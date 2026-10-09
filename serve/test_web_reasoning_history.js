// Run with Node. Exercise the web app's actual history serializer without a DOM.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const appFlag = process.argv.indexOf('--app');
const source = fs.readFileSync(appFlag >= 0 ? process.argv[appFlag + 1] : path.join(__dirname, 'web/app.js'), 'utf8');
const start = source.indexOf('function assistantMessages(m) {');
const end = source.indexOf('\nfunction setBusy(', start);
assert(start >= 0 && end > start);
const serialize = new Function(source.slice(start, end) + '\nreturn assistantMessages;')();
if (process.argv.includes('--serialize')) {
  process.stdout.write(JSON.stringify(serialize(JSON.parse(fs.readFileSync(0, 'utf8')))));
} else {
  assert.deepEqual(serialize({text:'The answer is 42.', reasoning:'Compute six times seven.'}),
    [{role:'assistant', content:'The answer is 42.', reasoning_content:'Compute six times seven.'}]);
  assert.deepEqual(serialize({text:'', reasoning:'Still working.'}),
    [{role:'assistant', content:'', reasoning_content:'Still working.'}]);
  assert.deepEqual(serialize({text:'Plain answer', reasoning:''}), [{role:'assistant', content:'Plain answer'}]);
  assert.deepEqual(serialize({text:'Interrupted', reasoning:'Keep this too.', stopped:true}),
    [{role:'assistant', content:'Interrupted', reasoning_content:'Keep this too.'}]);
  console.log('PASS: completed, interrupted, reasoning-only and plain assistant history');
}
