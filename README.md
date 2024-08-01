**lista dei `results` e breve descrizione:**
- `baseline-phimini-hotpotqa.csv`: risultati del processo tesi-antitesi-sintesi usando `microsoft/Phi-3-mini-4k-instruct` (3.8B) su questa versione `saracandu/hotpotQA_nli` del hotpotQA dataset;
- `baseline-phimedium-hotpotqa.csv`: risultati del processo tesi-antitesi-sintesi usando `microsoft/Phi-3-medium-4k-instruct` (14B) su questa versione `saracandu/hotpotQA_nli` del hotpotQA dataset;
- `bart-phimini-hotpotqa.csv`: tesi-antitesi-sintesi con come antitesi l'opzione tra `first_nli` e `second_nli` che ha lo score `BARTi` più alto;
- `roberta-phimini-hotpotqa.csv`: tesi-antitesi-sintesi con come antitesi l'opzione tra `first_nli` e `second_nli` che ha lo score `ROBERTAi` più alto;
- `bart-phimedium-hotpotqa.csv`: tesi-antitesi-sintesi con come ...;
- `roberta-phimedium-hotpotqa.csv`tesi-antitesi-sintesi con come ...;
- `baseline-llama3-instruct-hotpotqa.csv`: risultati del processo tesi-antitesi-sintesi usando `meta-llama/Meta-Llama-3.1-8B-Instruct` (8B) su questa versione `saracandu/hotpotQA_nli` del hotpotQA dataset;
- `baseline-gemma-2-2b-it-hotpotqa.csv`: tested on hotpotqa;
- `baseline-gemma-2-9b-it-hotpotqa.csv`: tested on hotpotqa.
 
**lista dei `py-files` e breve descrizione:**
- `baseline-phimini-hotpotqa.py`: file python che produce `baseline-phimini-hotpotqa.csv`;
- `bart-phimini-hotpotqa.py`: file python che auspicabilmente _produrrà_ `phi-mini-bart.csv`;
- `roberta-phimini-hotpotqa.py`: file python che auspicabilmente _produrrà_ `phi-mini-roberta.csv`;

**Note su `saracandu/hotpotQA_nli`:**
- `question`, `answer`, `passages`, `type`, `level` sono presi dal dataset originale;
- `options` è costruita in modo semi-automatico (con controllo mio a mano ex-post) prendendo `answer` e cercando un'altra opzione realistica a partire da `question` e `passages`;
- `first_nli` e `second_nli` sono statements creati da `microsoft/Phi-3-mini-4k-instruct` a partire da `question` e `options` con qualche esempio few-shot (tecnica suggerita dal paper https://arxiv.org/pdf/2104.08731 come alternative alla loro versione "rule-based)
- le coppie (`BART1`, `BART2`) e (`ROBERTA1`, `ROBERTA2`) sono ricavate rspettivamente dai modelli `facebook/bart-large-mnli` e `FacebookAI/roberta-large-mnli` come mostrato qui: https://github.com/gaoithee/tesi-bozze/blob/main/NLI-dataset.ipynb (è bruttissimo ma funzionale allo scopo).

**ATTUALMENTE IN RUN:**
1. `wikihop` dataset con le fonti sintetizzate da `facebook/bart-large-cnn`-> *nota importante: siccome ha un max di 4k in input, se il documento da sintetizzare è più grande viene brutalmente spezzato ogni 4k, sintetizzato indipendentemente ed infine questi riassunti vengono concatenati.* (poco raffinata ma...) 
2. `phi-mini-nocontext.csv`

**TO-DO:**
- sistema wikihop;
- togli il contesto e vedi cosa cambia;
- controlla `phi-small` (anche se ha un tokenizer diverso).
