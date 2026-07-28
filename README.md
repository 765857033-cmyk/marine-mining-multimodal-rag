# 鍩轰簬 Agentic RAG 鐨勬捣娲嬬熆浜х鐮旀枃鐚櫤鑳介棶绛旂郴缁?
杩欎釜椤圭洰瀹炵幇浜嗗浘鐗囪姹備腑鐨?Agent锛氶潰鍚戞捣娲嬬熆浜т笌娣辨捣澶氶噾灞炵粨鏍搞€佸瘜閽寸粨澹炽€佺儹娑茬～鍖栫墿绛夌鐮旇鏂囷紝鏀寔 PDF 鏂囨湰/琛ㄦ牸/鍥剧墖/鏁撮〉鎴浘瑙ｆ瀽銆佹枃鏈竻娲椼€乧hunk 鍒囧垎銆佸悜閲忓寲瀛樺偍銆佹贩鍚堟绱€丷erank銆佺瓟妗堢敓鎴愬拰鏉ユ簮杩芥函銆?
## 鎶€鏈爤

- Python
- LangChain / LangGraph
- 娣峰悎妫€绱細FAISS / numpy 鍚戦噺妫€绱?+ BM25 鍏抽敭璇嶆绱?+ RRF 铻嶅悎
- PyMuPDF
- Streamlit
- 澶氭ā鎬?PDF 瑙ｆ瀽锛氭枃鏈€佽〃鏍笺€佸浘鐗囥€侀〉闈㈡埅鍥撅紱閰嶇疆 OpenAI Key 鍚庡彲璋冪敤瑙嗚妯″瀷鐢熸垚鍥剧墖鎽樿
- Rerank锛氫紭鍏?CrossEncoder锛岀己澶辨椂浣跨敤棰嗗煙璇嶅拰 query overlap 瑙勫垯閲嶆帓
- Pandas

## 蹇€熷惎鍔?
```powershell
cd C:\Users\76585\Documents\Codex\2026-07-05\ne\outputs\mining-agentic-rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
streamlit run app.py
```

濡傛灉鍙兂鍏堣繍琛屽彲鐢ㄧ晫闈紝涔熷彲浠ユ墽琛岋細

```powershell
.\run.ps1
```

濡傞渶璋冪敤 OpenAI 鐢熸垚鏇磋嚜鐒剁殑绛旀锛屽湪 `.env` 涓厤缃細

```env
OPENAI_API_KEY=浣犵殑 key
OPENAI_MODEL=gpt-4o-mini
VISION_MODEL=gpt-4o-mini
```

娌℃湁 API Key 鏃讹紝绯荤粺浠嶅彲瀹屾垚 PDF 鍏ュ簱銆佽〃鏍兼娊鍙栥€佸浘鐗囦繚瀛樸€佸熀浜?caption/閭昏繎鏂囨湰鐨勮瑙夎瘉鎹储寮曘€佹贩鍚堟绱€侀噸鎺掑拰鎶藉彇寮忎腑鏂囧洖绛斻€傞厤缃?API Key 鍚庯紝鍥剧墖鍜岄〉闈㈡埅鍥句細璋冪敤瑙嗚妯″瀷鐢熸垚鏇村己鐨勫妯℃€佹憳瑕併€?
楂樺噯纭巼妯″紡寤鸿瀹夎棰濆渚濊禆骞跺惎鐢ㄥ璇█ embedding / cross-encoder rerank锛?
```powershell
.\install_accuracy_deps.ps1
```

鎺ㄨ崘 `.env`锛?
```env
EMBEDDING_BACKEND=sentence-transformers
SENTENCE_TRANSFORMER_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RERANK_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
```

## 鎺ュ叆 OceanGPT API

鏈」鐩敮鎸?OpenAI-compatible 鐨勮繙绋?OceanGPT 鏈嶅姟銆傞€傚悎鎶?OceanGPT 閮ㄧ讲鍦?GPU 鏈嶅姟鍣ㄣ€丄utoDL銆乿LLM銆丼GLang 鎴?LMDeploy 涓婏紝鏈満鍙礋璐ｆ枃妗ｈВ鏋愩€佹绱㈠拰璇佹嵁鏍￠獙銆?
鍦?`.env` 涓厤缃細

```env
LLM_BACKEND=ocean-gpt-api
OCEANGPT_API_KEY=浣犵殑鏈嶅姟瀵嗛挜
OCEANGPT_BASE_URL=https://浣犵殑鏈嶅姟鍦板潃/v1
OCEANGPT_MODEL=浣犵殑妯″瀷鍚?LLM_TIMEOUT=120
```

濡傛灉浣犵殑鏈嶅姟涓嶆槸 OceanGPT锛屼絾鍏煎 OpenAI `/v1/chat/completions`锛屽彲浠ョ敤锛?
```env
LLM_BACKEND=openai-compatible
CUSTOM_LLM_API_KEY=浣犵殑鏈嶅姟瀵嗛挜
CUSTOM_LLM_BASE_URL=https://浣犵殑鏈嶅姟鍦板潃/v1
CUSTOM_LLM_MODEL=浣犵殑妯″瀷鍚?```

鍥炵瓟浠嶄細琚郴缁熸彁绀鸿瘝绾︽潫涓轰腑鏂囷紝骞朵笖鍙熀浜庢绱㈠埌鐨勬潵婧愮墖娈靛洖绛斻€?
## LLM Router 涓庝笂涓嬫枃璁板繂

寮€鍚?`LLM_ROUTER_ENABLED=true` 鍚庯紝绯荤粺浼氫紭鍏堣宸查厤缃殑澶фā鍨嬪垽鏂槸鍚﹂渶瑕佽皟鐢ㄥ閮ㄨ鏂囩煡璇嗗簱銆俁outer 浼氳緭鍑?`rag/direct`銆佹绱㈡煡璇€佸師鍥犲拰缃俊搴︼紱濡傛灉鏈厤缃彲鐢ㄦā鍨嬶紝鍒欒嚜鍔ㄩ檷绾т负瑙勫垯璺敱銆?
```env
LLM_ROUTER_ENABLED=true
MEMORY_ENABLED=true
MEMORY_MAX_TURNS=6
```

Streamlit 渚ц竟鏍忓彲浠ユ墦寮€鎴栧叧闂€滃惎鐢ㄥぇ妯″瀷璺敱鈥濆拰鈥滃惎鐢ㄤ笂涓嬫枃璁板繂鈥濄€備笂涓嬫枃璁板繂浼氭妸鏈€杩戝嚑杞敤鎴烽棶棰樺拰鍔╂墜鍥炵瓟浼犵粰 Router 涓庣瓟妗堢敓鎴愯妭鐐癸紝鐢ㄤ簬澶勭悊鈥滅户缁垰鎵嶇殑闂鈥濃€滆繖浜涘洜绱犫€濈瓑杩介棶銆?
## 鐖跺瓙鏂囨。鍒囩墖

寮€鍚?`PARENT_CHILD_ENABLED=true` 鍚庯紝鍏ュ簱浼氬厛鐢熸垚杈冨ぇ鐨勭埗鏂囨。鍧楋紝鍐嶄粠鐖舵枃妗ｄ腑鍒囧嚭杈冨皬鐨勫瓙鏂囨。鍧椼€傚悜閲忔绱㈠拰 BM25 妫€绱㈠彧绱㈠紩瀛愭枃妗ｅ潡锛屽懡涓悗鍥炲彫瀹屾暣鐖舵枃妗ｄ笂涓嬫枃杩涘叆 rerank 鍜岀瓟妗堢敓鎴愩€?
```env
PARENT_CHILD_ENABLED=true
PARENT_CHUNK_SIZE=1800
PARENT_CHUNK_OVERLAP=200
CHILD_CHUNK_SIZE=420
CHILD_CHUNK_OVERLAP=80
```

妫€绱㈡祦绋嬶細

```text
query 鈫?child chunks 绮惧噯鍙洖 鈫?parent_id 鍥炲彫鐖舵枃妗?鈫?rerank 鈫?answer_generation
```

Streamlit 鐨勬潵婧愬睍寮€涓細鏄剧ず鈥滃懡涓殑瀛愮墖娈碘€濆拰鈥滃洖鍙殑鐖舵枃妗ｄ笂涓嬫枃鈥濄€?
## FastAPI 鏈嶅姟

椤圭洰鐜板湪鎻愪緵 FastAPI 鍚庣锛屽彲浠ユ妸 Streamlit Demo 鍗囩骇涓哄彲琚墠绔€佽剼鏈€佽瘎娴嬪钩鍙版垨浼佷笟绯荤粺璋冪敤鐨?RAG 鏈嶅姟銆?
鍚姩锛?
```powershell
.\run_api.ps1
```

榛樿鍦板潃锛?
```text
http://127.0.0.1:8000
http://127.0.0.1:8000/docs
```

涓昏鎺ュ彛锛?
```text
GET  /health             鍋ュ悍妫€鏌?GET  /index/stats        鏌ョ湅绱㈠紩鐘舵€併€乧hunk 鏁般€佹潵婧愬拰妯℃€佸垎甯?POST /documents/upload   涓婁紶 PDF锛岃В鏋愬苟閲嶅缓绱㈠紩
POST /index/rebuild      浣跨敤宸蹭笂浼?PDF 鎴栨寚瀹氳矾寰勯噸寤虹储寮?POST /chat               璋冪敤 Agentic RAG 闂瓟
```

FastAPI 灞傚鐢?`AgentFastAPI API`銆丮inerU/PyMuPDF/Docling 瑙ｆ瀽銆佺埗瀛愭枃妗ｅ垏鐗囥€佹贩鍚堟绱€丷RF銆乺erank 鍜?source verification锛屼笉缁存姢鍙︿竴濂楅棶绛旈€昏緫銆?
## 浣跨敤 MinerU 瑙ｆ瀽 PDF

MinerU 鐜板湪鏄」鐩粯璁ょ殑 PDF 缁撴瀯鍖栬В鏋愬悗绔紝閫傚悎绉戠爺璁烘枃銆佸鏉傜増闈€佽〃鏍笺€佸叕寮忋€佸浘娉ㄥ拰鍥炬枃娣锋帓鏂囨。銆傚畨瑁咃細

```powershell
.\install_mineru.ps1
```

鐒跺悗鍦?Streamlit 渚ц竟鏍忔妸 `PDF 瑙ｆ瀽鍚庣` 鍒囨崲涓?`mineru`锛屾垨鍦?`.env` 涓缃細

```env
PARSER_BACKEND=mineru
PARSER_FALLBACK=true
MINERU_COMMAND=mineru
MINERU_BACKEND=pipeline
MINERU_METHOD=auto
MINERU_TIMEOUT=900
MINERU_EXPORT_ARTIFACTS=true
```

鍏ュ簱鏃剁郴缁熶細璋冪敤 MinerU CLI锛屾妸 PDF 瑙ｆ瀽鎴?Markdown / JSON 缁撴瀯鍖栦骇鐗╋紝浼樺厛璇诲彇 `content_list` 涓殑鏂囨湰鍧椼€佽〃鏍煎潡銆佸浘鐗?鍥炬敞鍧楀拰鍏紡鍧楋紱濡傛灉 JSON 涓嶅瓨鍦紝鍒欏洖閫€璇诲彇 MinerU 瀵煎嚭鐨?Markdown銆傝В鏋愬悗鐨勮瘉鎹細缁熶竴杩涘叆鐖跺瓙鏂囨。鍒囩墖銆佹贩鍚堟绱€丷RF銆乺erank 鍜?source verification 娴佺▼銆侻inerU 浜х墿榛樿淇濆瓨鍦?`data/uploads/_mineru/<pdf-name>/`銆?
`MINERU_BACKEND=pipeline` 鏇撮€傚悎鏈満 CPU/鏅€?GPU 鐜锛涘鏋滃悗缁鎺ヨ瑙夎瑷€妯″瀷鐗?MinerU锛屽彲浠ユ妸渚ц竟鏍忕殑 backend 鍒囧埌 `vlm-engine`銆乣hybrid-engine` 鎴?HTTP client 妯″紡锛屼絾浼氭洿鍚冩樉瀛樻垨渚濊禆澶栭儴 MinerU 鏈嶅姟銆?
## 浣跨敤 Docling 瑙ｆ瀽 PDF

Docling 鍙互浣滀负楂樿川閲?PDF 瑙ｆ瀽鍚庣锛岀敤浜庢洿濂界殑鐗堥潰鐞嗚В銆侀槄璇婚『搴忋€佽〃鏍肩粨鏋勫拰 RAG chunk銆傚畨瑁咃細

```powershell
.\install_docling.ps1
```

鐒跺悗鍦?Streamlit 渚ц竟鏍忔妸 `PDF 瑙ｆ瀽鍚庣` 鍒囨崲涓?`docling`锛屾垨鍦?`.env` 涓缃細

```env
PARSER_BACKEND=docling
PARSER_FALLBACK=true
DOCLING_USE_HYBRID_CHUNKER=true
DOCLING_DO_TABLE_STRUCTURE=true
DOCLING_DO_OCR=false
DOCLING_EXPORT_ARTIFACTS=true
```

Docling 浼氫紭鍏堜娇鐢?`DocumentConverter` 鍜?`HybridChunker` 鐢熸垚涓婁笅鏂囧寲鏂囨。鐗囨锛屽苟鎶?Docling 琛ㄦ牸杞垚 `table` 妯℃€佽瘉鎹€傚鍑虹殑 Markdown/JSON 浼氫繚瀛樺湪 `data/uploads/_docling/<pdf-name>/`锛屽悗缁粛鐒惰繘鍏ユ湰椤圭洰宸叉湁鐨勫鏌ヨ娣峰悎妫€绱€乺erank 鍜?source verification 鑺傜偣銆?
## 鍑嗙‘鐜囪瘎娴?
瑕佽瘉鏄庢槸鍚﹁揪鍒?90% 浠ヤ笂锛岄渶瑕佸噯澶囧甫鏍囧噯绛旀鎴栨湡鏈涜瘉鎹殑璇勬祴闆嗐€傚厛鍦?Streamlit 涓畬鎴?PDF 鍏ュ簱锛岀劧鍚庣紪杈?`evaluation/sample_eval.jsonl`锛屼负姣忎釜闂濉啓 `expected_terms` 鍜屽彲閫夌殑 `expected_sources`銆?
```powershell
.\.venv\Scripts\python.exe tools\evaluate.py --cases evaluation\sample_eval.jsonl
```

杈撳嚭浼氬寘鍚?accuracy銆乻ource hit rate銆乤verage term recall 鍜?verification pass rate銆傚缓璁嚦灏戝噯澶?50-100 涓湡瀹炵鐮旈棶棰樺啀鍒ゆ柇绯荤粺鏄惁杈惧埌 90%銆?
## Agent 宸ヤ綔娴?
1. 鐢ㄦ埛闂杩涘叆 `question_analysis/router` 鑺傜偣銆?2. Router 鍒ゆ柇鏄惁闇€瑕佸閮ㄦ枃鐚煡璇嗗簱銆?3. 鏃犻渶妫€绱㈡椂杩涘叆 `direct_answer` 鑺傜偣骞剁洿鎺ヨ緭鍑恒€?4. 闇€瑕佹绱㈡椂杩涘叆 `query_rewrite` 鑺傜偣锛屼负澶氶噾灞炵粨鏍搞€佸瘜閽寸粨澹炽€佺儹娑茬～鍖栫墿绛夋湳璇ˉ鍏呰嫳鏂囧悓涔夎瘝鍜岀鐮旀绱㈣瘝銆?5. 鍏ュ簱闃舵宸茬粡鎶婃枃鏈€佽〃鏍笺€佸浘鐗囨憳瑕佸拰椤甸潰鎴浘鎽樿缁熶竴杞崲涓哄彲妫€绱㈣瘉鎹€?6. 杩涘叆 `hybrid_retrieval` 鑺傜偣锛屽悓鏃舵墽琛屽悜閲忔绱㈠拰 BM25 鍏抽敭璇嶆绱紝骞朵娇鐢?RRF 铻嶅悎鍊欓€夌墖娈点€?7. 杩涘叆 `rerank` 鑺傜偣锛岀患鍚堝悜閲忓垎鏁般€佸叧閿瘝瑕嗙洊銆侀鍩熻瘝鍛戒腑鎴?CrossEncoder 鍒嗘暟閲嶆柊鎺掑簭銆?8. 杩涘叆 `answer_generation` 鑺傜偣锛屽熀浜庢枃鏈?琛ㄦ牸/瑙嗚璇佹嵁鐢熸垚甯︽潵婧愬洖绛斻€?9. 杩涘叆 `source_verification` 鑺傜偣鍒ゆ柇璇佹嵁鏄惁鍏呭垎銆?10. 濡傛灉璇佹嵁涓嶈冻锛屽苟涓旀湭杈惧埌鏈€澶ф绱㈣疆娆★紝鍒欏洖鍒?`query_rewrite` 鑺傜偣鎵╁睍鏌ヨ骞朵簩娆℃绱€?11. 鏈€缁堣緭鍑虹瓟妗堛€乻ource 鏂囨。鍚嶃€侀〉鐮併€佹ā鎬併€佺墖娈靛拰妫€绱?閲嶆帓杞ㄨ抗銆?
```text
鐢ㄦ埛闂
  鈫?question_analysis / router
  鈫?鏄惁闇€瑕佸閮ㄧ煡璇嗗簱锛?  鈹溾攢 鍚?鈫?direct_answer 鈫?鏈€缁堝洖绛?  鈹斺攢 鏄?鈫?query_rewrite
            鈫?         hybrid_retrieval(vector + BM25 + RRF)
            鈫?         rerank
            鈫?         answer_generation
            鈫?         source_verification
            鈫?      璇佹嵁鍏呭垎锛?鈹€ 鏄?鈫?鏈€缁堝洖绛?+ 鏉ユ簮
            鈫?鍚?         query_rewrite 鈫?鍐嶆绱?```

## 鐩綍

```text
app.py                  Streamlit 鐣岄潰
src/agent.py            LangGraph Agentic RAG 瀹屾暣鑺傜偣娴?src/api.py              FastAPI 鏂囨。涓婁紶銆佺储寮曟瀯寤恒€侀棶绛斿拰鐘舵€佹帴鍙?src/pdf_ingest.py       PyMuPDF PDF 瑙ｆ瀽
src/mineru_ingest.py    MinerU CLI / content_list / Markdown 瑙ｆ瀽鍚庣
src/docling_ingest.py   Docling DocumentConverter / HybridChunker 瑙ｆ瀽鍚庣
src/multimodal.py       琛ㄦ牸銆佸浘鐗囥€佹暣椤垫埅鍥惧拰瑙嗚鎽樿瑙ｆ瀽
src/text_processing.py  鏂囨湰娓呮礂銆佸幓椤电湁椤佃剼銆佸幓鍙傝€冩枃鐚€乧hunk 鍒囧垎
src/vector_store.py     embedding銆丗AISS/numpy 鍚戦噺绱㈠紩銆丅M25 鍜?RRF 娣峰悎妫€绱?src/rerank.py           CrossEncoder 鎴栬鍒?Rerank
src/llm.py              OpenAI 鎴栨娊鍙栧紡绛旀鐢熸垚
tools/evaluate.py       妫€绱㈣瘉鎹噯纭巼璇勬祴鑴氭湰
```

## 瀹屾暣鐗堣瘎浠蜂綋绯?
璇勬祴鑴氭湰宸茬粡鍗囩骇涓烘绱㈠眰銆佺敓鎴愬眰鍜岀郴缁熷眰涓夌被鎸囨爣锛?
```text
妫€绱㈠眰锛欻it@K銆丮RR銆乶DCG銆丼ource Hit Rate銆乀erm Recall
鐢熸垚灞傦細Citation Accuracy銆丗aithfulness銆丄nswer Term Recall
绯荤粺灞傦細Refusal Accuracy銆乂erification Pass Rate銆丷etrieval Rounds銆丩atency銆佸彲閫?LLM-as-a-Judge
```

杩愯锛?
```powershell
.\.venv\Scripts\python.exe tools\evaluate.py --cases evaluation\sample_eval.jsonl --hit-ks 1,3,5
```

淇濆瓨 JSON 缁撴灉锛?
```powershell
.\.venv\Scripts\python.exe tools\evaluate.py `
  --cases evaluation\sample_eval.jsonl `
  --hit-ks 1,3,5,10 `
  --output evaluation\eval_result.json
```

鍚敤 LLM Judge锛岄渶瑕佸厛閰嶇疆鍙敤鐨勫ぇ妯″瀷 API锛?
```powershell
.\.venv\Scripts\python.exe tools\evaluate.py --cases evaluation\sample_eval.jsonl --llm-judge
```

璇勬祴闆?JSONL 鏀寔鏃ф牸寮忥紝涔熸敮鎸佹洿绮剧‘鐨?source/page 鏍囨敞锛?
```json
{
  "question": "澶氶噾灞炵粨鏍告垚鐭垮彈鍝簺鐜鍥犵礌鎺у埗锛?,
  "expected_sources": [{"source": "paper.pdf", "page": 5}],
  "expected_terms": ["姘у寲杩樺師", "娌夌Н閫熺巼", "閲戝睘鏉ユ簮"],
  "min_term_recall": 0.6,
  "should_refuse": false
}
```

濡傛灉鏄煡璇嗗簱涓病鏈夌瓟妗堢殑闂锛屽彲浠ユ爣娉細

```json
{
  "question": "杩欐壒鏂囨。鏄惁鍖呭惈鏌愪釜涓嶅瓨鍦ㄧ殑鎸囨爣锛?,
  "expected_terms": [],
  "should_refuse": true
}
```

