"""Validate this checkpoint's uniform per-layer activation scales and save 48 FP32 pairs."""
import argparse
from pathlib import Path
import numpy as np
from gguf import GGUFReader

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    tensors = {t.name: t for t in GGUFReader(args.gguf).tensors}
    values = []
    for layer in range(48):
        scales = []
        for projection in ('gate', 'up', 'down'):
            name = f'blk.{layer}.ffn_{projection}_exps.input_scale'
            tensor = tensors.get(name)
            if tensor is None:
                raise ValueError(f'Missing activation scales: {name}')
            a = np.asarray(tensor.data, dtype=np.float32).reshape(-1)
            if a.size != 512 or not np.all(a == a[0]) or not np.isfinite(a[0]) or a[0] <= 0:
                raise ValueError(f'{name}: expected 512 identical positive finite scales')
            scales.append(a[0])
        if scales[0] != scales[1]:
            raise ValueError(f'Layer {layer}: gate and up activation scales differ')
        values.extend((scales[0], scales[2]))
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(values, dtype='<f4').tofile(destination)
    if destination.stat().st_size != 384:
        raise OSError('Incomplete scale output')
    print(f'Validated 73728 activation scales; wrote 48 gate/up,down pairs to {destination}')

if __name__ == '__main__':
    main()
