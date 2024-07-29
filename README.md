**lista dei files e breve descrizione:**
- `phi-mini-baseline.csv`: risultati del processo tesi-antitesi-sintesi usando `microsoft/Phi-3-mini-4k-instruct` (3.8B) su questa versione `saracandu/hotpotQA_nli` del hotpotQA dataset;
- `phi-medium-baseline.csv`: risultati del processo tesi-antitesi-sintesi usando `microsoft/Phi-3-medium-4k-instruct` (14B) su questa versione `saracandu/hotpotQA_nli` del hotpotQA dataset;


**Note su `saracandu/hotpotQA_nli`:**
- `question`, `answer`, `passages`, `type`, `level` sono presi dal dataset originale;
- `options` è costruita in modo semi-automatico (con controllo mio a mano ex-post) prendendo `answer` e cercando un'altra opzione realistica a partire da `question` e `passages`;
- `first_nli` e `second_nli` sono statements creati da `microsoft/Phi-3-mini-4k-instruct` a partire da `question` e `options` con qualche esempio few-shot (tecnica suggerita dal paper https://arxiv.org/pdf/2104.08731 come alternative alla loro versione "rule-based);
- le coppie (`BART1`, `BART2`) e (`ROBERTA1`, `ROBERTA2`) sono ricavate rispettivamente dai modelli `facebook/bart-large-mnli` e `FacebookAI/roberta-large-mnli` come mostrato qui: https://github.com/gaoithee/tesi-bozze/blob/main/NLI-dataset.ipynb (è bruttissimo ma funzionale allo scopo).


**ATTUALMENTE IN RUN:**
1. `phi-mini-bart.csv`: tesi-antitesi-sintesi con come antitesi l'opzione tra `first_nli` e `second_nli` che ha lo score `BARTi` più alto;
2. `phi-mini-roberta.csv`: tesi-antitesi-sintesi con come antitesi l'opzione tra `first_nli` e `second_nli` che ha lo score `ROBERTAi` più alto;
3. `wikihop` dataset con le fonti sintetizzate da `facebook/bart-large-cnn` (minimo numero di token: 10, massimo: 400) -> *valutare se estenderlo e se sì, di quanto!*
4. `llama3-instruct-baseline.csv`: risultati del processo tesi-antitesi-sintesi usando `meta-llama/Meta-Llama-3.1-8B-Instruct` (8B) su questa versione `saracandu/hotpotQA_nli` del hotpotQA dataset (devo tagliare esempi perché ovviamente l'inference time è più alto, per ora `N_examples = 200`);
