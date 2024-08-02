**`results` and short description:**

on `hotpotqa` dataset:
|which model considered |baseline♦︎ + oracle✦|bart✸|roberta✱| no context | cot |
|:------:|:---:|:---:|:---:|:---:|:---:|
| phi-mini | `baseline-phimini-hotpotqa.csv`| `bart-phimini-hotpotqa.csv` | `roberta-phimini-hotpotqa.csv` | `nocontext-phimini-hotpotqa.csv` | `cot-phimini-hotpotqa.csv` |
| phi-medium | `baseline-phimedium-hotpotqa.csv` | `bart-phimedium-hotpotqa.csv` | `roberta-phimedium-hotpotqa.csv` | `nocontext-phimedium-hotpotqa.csv` | / |
| llama-3.1-it-8b | `baseline-llama-3.1-instruct-hotpotqa.csv` | / | / | / | / |
| gemma-2-2b-it | `baseline-gemma-2-2b-it-hotpotqa.csv` | / | / | / | / |
| gemma-2-9b-it | `baseline-gemma-2-2b-it-hotpotqa.csv` | / | / | / | / |

baseline (♦︎) means that the LLM produces the antithesis given the thesis answer and the context;
oracle (✦) means that the antithesis is the correct suggestion;
bart (✸) means that the antithesis is the NLI statement with the highest entailment probability according to `facebook/bart-large`;
roberta (✱) means same as above but with `FacebookAI/roberta-large`.
 
**`py-files`**
on `hotpotqa` dataset:

|which model considered |baseline♦︎ + oracle✦|bart✸|roberta✱|
|:------:|:---:|:---:|:---:|
| phi-mini | `baseline-phimini-hotpotqa.py`| `bart-phimini-hotpotqa.py` | `roberta-phimini-hotpotqa.py` |
| phi-medium | / | `bart-phimedium-hotpotqa.py` | `roberta-phimedium-hotpotqa.py` |
| llama-3.1-it-8b | `baseline-llama-3.1-instruct-hotpotqa.py` | / | / |
| gemma-2-2b-it | `baseline-gemma-2-2b-it-hotpotqa.py` | / | / |
| gemma-2-9b-it | `baseline-gemma-2-2b-it-hotpotqa.py` | / | / | 

_____

**DATASETS**
**`saracandu/hotpotQA_nli`:**
- `question`, `answer`, `passages`, `type`, `level` sono presi dal dataset originale;
- `options` è costruita in modo semi-automatico (con controllo mio a mano ex-post) prendendo `answer` e cercando un'altra opzione realistica a partire da `question` e `passages`;
- `first_nli` e `second_nli` sono statements creati da `microsoft/Phi-3-mini-4k-instruct` a partire da `question` e `options` con qualche esempio few-shot (tecnica suggerita dal paper https://arxiv.org/pdf/2104.08731 come alternative alla loro versione "rule-based)
- le coppie (`BART1`, `BART2`) e (`ROBERTA1`, `ROBERTA2`) sono ricavate rspettivamente dai modelli `facebook/bart-large-mnli` e `FacebookAI/roberta-large-mnli` come mostrato qui: https://github.com/gaoithee/tesi-bozze/blob/main/NLI-dataset.ipynb (è bruttissimo ma funzionale allo scopo).

**ATTUALMENTE IN RUN:**
- `gemma-2-27b-antithesis`;
- `phimedium-cot`;
- `phimini-pure-cot`;
- `phimedium-pure-cot`.

**TO-DO:**
- sistema `wikihop` dataset con le fonti sintetizzate da `facebook/bart-large-cnn`-> *nota importante: siccome ha un max di 4k in input, se il documento da sintetizzare è più grande viene brutalmente spezzato ogni 4k, sintetizzato indipendentemente ed infine questi riassunti vengono concatenati.* (poco raffinata ma...);
- controlla `phi-small` (anche se ha un tokenizer diverso) e struttura non equivalente;
- fai il test `no context` su `llama` e i vari gemma;
- compara `llama` e i vari `gemma` con `pure-cot`;
- valuta se testare `bart`, `roberta`, `cot` anche su `llama` e i vari `gemma`.
