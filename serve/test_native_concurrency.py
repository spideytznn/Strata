"""The native engine's shared M-RoPE table must never cross concurrent requests."""
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from serve.server import StrataEngine


class NativeVisionGateTests(unittest.TestCase):
    def setUp(self):
        self.engine = object.__new__(StrataEngine)
        self.engine.batch = 2
        self.engine.info = {"weight_source": "safetensors"}
        self.engine.vision_cv = threading.Condition()
        self.engine.slot_cv = threading.Condition()
        self.engine.vision_readers = 0
        self.engine.vision_writer = False
        self.engine.vision_waiting = 0
        self.engine.slot_busy = [False, False]
        self.engine.alive = lambda: True
        self.started = []
        self.releases = {name: threading.Event() for name in ("a", "b", "image", "late")}

        def generate(ids, max_new, sampling, cancel, embeddings=None):
            self.started.append(ids)
            yield 42
            while not self.releases[ids].wait(0.01) and not cancel.is_set():
                yield None

        self.engine.generate_batched = generate
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.addCleanup(self.pool.shutdown)
        self.addCleanup(lambda: [ev.set() for ev in self.releases.values()])

    def run_request(self, name, image=False, cancel=None):
        return self.pool.submit(lambda: list(self.engine.generate(
            name, 32, {}, cancel or threading.Event(), "image-file" if image else None)))

    def until(self, predicate):
        end = time.monotonic() + 3
        while not predicate() and time.monotonic() < end:
            time.sleep(0.005)
        self.assertTrue(predicate())

    def test_text_parallel_image_exclusive_writer_preference(self):
        a = self.run_request("a")
        b = self.run_request("b")
        self.until(lambda: len(self.started) == 2)
        image = self.run_request("image", image=True)
        self.until(lambda: self.engine.vision_waiting == 1)
        late = self.run_request("late")
        self.releases["a"].set()
        a.result(timeout=3)
        self.assertNotIn("image", self.started)
        self.releases["b"].set()
        b.result(timeout=3)
        self.until(lambda: "image" in self.started)
        self.assertNotIn("late", self.started)
        self.releases["image"].set()
        image.result(timeout=3)
        self.until(lambda: "late" in self.started)
        self.releases["late"].set()
        late.result(timeout=3)
        self.assertEqual(self.engine.vision_readers, 0)
        self.assertFalse(self.engine.vision_writer)

    def test_cancel_waiting_image_does_not_block_new_text(self):
        a = self.run_request("a")
        self.until(lambda: "a" in self.started)
        cancel = threading.Event()
        image = self.run_request("image", image=True, cancel=cancel)
        self.until(lambda: self.engine.vision_waiting == 1)
        cancel.set()
        image.result(timeout=3)
        self.assertNotIn("image", self.started)
        b = self.run_request("b")
        self.until(lambda: "b" in self.started)
        self.releases["a"].set()
        self.releases["b"].set()
        a.result(timeout=3)
        b.result(timeout=3)

    def test_image_waits_for_background_slot_drain(self):
        self.engine.slot_busy[0] = True
        image = self.run_request("image", image=True)
        self.until(lambda: self.engine.vision_writer)
        self.assertNotIn("image", self.started)
        with self.engine.slot_cv:
            self.engine.slot_busy[0] = False
            self.engine.slot_cv.notify_all()
        self.until(lambda: "image" in self.started)
        self.releases["image"].set()
        image.result(timeout=3)

    def test_closed_generator_releases_permit(self):
        gen = self.engine.generate("a", 32, {}, threading.Event())
        self.assertEqual(next(gen), 42)
        gen.close()
        self.assertEqual(self.engine.vision_readers, 0)

    def test_closed_image_drain_releases_writer(self):
        self.engine.slot_busy[0] = True
        gen = self.engine.generate("image", 32, {}, threading.Event(), "image-file")
        self.assertIsNone(next(gen))
        self.assertTrue(self.engine.vision_writer)
        gen.close()
        self.assertFalse(self.engine.vision_writer)
        self.assertEqual(self.engine.vision_waiting, 0)


if __name__ == "__main__":
    unittest.main()
