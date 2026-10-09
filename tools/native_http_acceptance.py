"""Real HTTP surface acceptance around an already-loaded private engine."""
import json,urllib.request,urllib.error
from serve.server import Service,serve

def validate_http(engine,tokenizer,template,output):
 svc=Service(engine,tokenizer,template,model_name='native-test')
 httpd=serve(svc,host='127.0.0.1',port=0)
 base=f'http://127.0.0.1:{httpd.server_address[1]}'
 result={'url':base,'checks':[]}
 def request(path,body=None,expect=200):
  req=urllib.request.Request(base+path,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
  try:
   with urllib.request.urlopen(req,timeout=300) as r:status=r.status;raw=r.read().decode()
  except urllib.error.HTTPError as e:status=e.code;raw=e.read().decode()
  row={'path':path,'status':status,'response':raw};result['checks'].append(row)
  output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
  assert status==expect,(path,status,raw)
  return raw
 try:
  request('/health');models=json.loads(request('/v1/models'));assert models['data']
  common={'model':'native-test','temperature':0,'max_tokens':32,'chat_template_kwargs':{'enable_thinking':False}}
  body={**common,'messages':[{'role':'user','content':'Reply with exactly Blue.'}]}
  answer=json.loads(request('/v1/chat/completions',body));assert 'Blue' in answer['choices'][0]['message']['content']
  stream=request('/v1/chat/completions',{**body,'stream':True});assert '[DONE]' in stream
  events=[json.loads(l[6:]) for l in stream.splitlines() if l.startswith('data: ') and l!='data: [DONE]']
  text=''.join(e['choices'][0].get('delta',{}).get('content') or '' for e in events if e.get('choices'))
  assert text==answer['choices'][0]['message']['content'],(text,answer)
  anthropic=json.loads(request('/v1/messages',{**common,'thinking':{'type':'disabled'},'messages':[{'role':'user','content':'Reply with exactly Green.'}]}))
  assert 'Green' in ''.join(x.get('text','') for x in anthropic['content'])
  tools=[{'type':'function','function':{'name':'lookup_code','description':'Look up the stored access code.',
          'parameters':{'type':'object','properties':{'name':{'type':'string'}},'required':['name'],'additionalProperties':False}}}]
  messages=[{'role':'user','content':'Use lookup_code to fetch the access code for Alice.'}]
  call=json.loads(request('/v1/chat/completions',{**common,'max_tokens':96,'messages':messages,'tools':tools,
                  'tool_choice':{'type':'function','function':{'name':'lookup_code'}}}))
  assistant=call['choices'][0]['message'];tc=assistant['tool_calls'][0]
  assert tc['function']['name']=='lookup_code';json.loads(tc['function']['arguments'])
  messages += [assistant,{'role':'tool','tool_call_id':tc['id'],'content':'{"code":"ZEBRA-417"}'},
               {'role':'user','content':'What is the code? Reply with the code only.'}]
  follow=json.loads(request('/v1/chat/completions',{**common,'messages':messages,'tools':tools,'tool_choice':'none'}))
  assert 'ZEBRA-417' in follow['choices'][0]['message']['content']
  thinking={**common,'max_tokens':192,'reasoning_effort':'low',
            'chat_template_kwargs':{'enable_thinking':True,'preserve_reasoning':True}}
  history=[{'role':'user','content':'Compute 3 + 4. Put only the number in the final answer.'}]
  first=json.loads(request('/v1/chat/completions',{**thinking,'messages':history}))
  message=first['choices'][0]['message']
  assert message.get('reasoning_content') and message.get('content','').strip()=='7',first
  history += [message,{'role':'user','content':'Multiply that result by 2. Put only the number in the final answer.'}]
  second=json.loads(request('/v1/chat/completions',{**thinking,'messages':history}))
  assert second['choices'][0]['message'].get('content','').strip()=='14',second
  reused=second['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0)
  expected_prefix=first['usage']['prompt_tokens']+first['usage']['completion_tokens']
  assert reused>=expected_prefix-8,('reasoning/answer prefix was not preserved',reused,expected_prefix)
  result['reasoning_history']={'previous_tokens':expected_prefix,'cached_tokens':reused}
  # froggeric v22.5 explicitly renders later system messages; ensure they are
  # retained by the HTTP normalization as well, rather than silently dropped.
  late=json.loads(request('/v1/chat/completions',{**common,'messages':[
    {'role':'user','content':'Remember the instructions that follow.'},
    {'role':'system','content':'The secret code is SIGMA-682.'},
    {'role':'user','content':'What is the secret code? Reply with the code only.'}]}))
  assert 'SIGMA-682' in late['choices'][0]['message']['content']
  request('/v1/chat/completions',{**common,'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,AA=='}}]}]},expect=400)
  request('/v1/chat/completions',{**body,'max_tokens':engine.max_context+1},expect=400)
  result['status']='pass';output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 finally:httpd.shutdown();httpd.server_close()
 return result
