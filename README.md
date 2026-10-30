# Imagem para texto

Aplicação de OCR com API FastAPI e interface React. O reconhecedor base é RapidOCR/PP-OCRv4. O piloto fine-tuned está disponível como alternativa explícita; o original continua a ser o modelo padrão porque a avaliação não demonstrou melhoria no conjunto de teste.

## Requisitos

- Python 3.10 ou superior
- Node.js 18 ou superior e npm
- Modelos do RapidOCR, instalados automaticamente com a dependência Python

## Instalação e execução (Windows PowerShell)

Na raiz do projeto, crie e ative o ambiente Python e instale as dependências:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

Inicie o backend num terminal:

```powershell
uvicorn backend.app.main:app --reload
```

Noutro terminal, instale e inicie o frontend:

```powershell
cd frontend
npm install
npm run dev
```

Abra `http://127.0.0.1:5173`. O frontend encaminha os pedidos `/api` para `http://127.0.0.1:8000`. Para outra origem, defina `VITE_API_BASE_URL` no ambiente de build do frontend e `FRONTEND_ORIGINS` no backend, separados por vírgulas.

## API

- `GET /api/health` devolve `{"status":"ok"}`.
- `POST /api/ocr` recebe `multipart/form-data` com o campo `image` (JPEG, PNG ou WEBP; máximo 10 MB e 25 megapíxeis) e devolve `text`, `lines` com pontuações OCR, `engine` e `processing_time_ms`.

## Testes e build

Na raiz, execute os testes do backend — incluindo o processamento OCR de cada imagem em `data/ocr/test/`:

```powershell
python -m pytest backend\tests -q
```

Valide o frontend:

```powershell
cd frontend
npm run build
```

## Fine-tuning piloto do reconhecedor

O treino parte do reconhecedor `ch_PP-OCRv4_rec_infer.onnx` incluído no pacote RapidOCR. O grafo ONNX é convertido para PyTorch com `onnx2torch`; a conversão é verificada numericamente contra o modelo original antes do treino. Apenas a matriz de projeção CTC pré-treinada é atualizada; o extrator visual fica congelado. Não existem caixas de texto anotadas, por isso o detetor não é afinado.

As caixas de linha em `data/ocr/annotations.csv` foram anotadas manualmente a partir de texto visível com transcrições verificadas. Três linhas da imagem de Londres são treino; duas linhas da imagem neon estão numa pasta e divisão de validação separadas, sem partilhar imagem-fonte com o treino. `data/ocr/test/` não é lido pelo script de treino.

Instala as dependências no ambiente Python ativo:

```powershell
pip install -r backend\requirements.txt -r training\requirements.txt
```

Executa 20 épocas reproduzíveis (seed/configuração em `training/config.json`):

```powershell
python training\fine_tune_recognizer.py
```

Os checkpoints por época, o melhor checkpoint, os logs, as métricas de validação e o ONNX exportado ficam em `artifacts/ocr_finetune/`. O modelo exportado usa uma imagem por lote; o backend já configura esta limitação. Compara os dois modelos no conjunto final reservado:

```powershell
python training\evaluate_holdout.py --model original
python training\evaluate_holdout.py --model finetuned
```

As avaliações ficam em `artifacts/ocr_finetune/evaluation-*.json`. Para experimentar o modelo afinado, escolhe “Fine-tuned · piloto” na interface ou envia `POST /api/ocr?model=finetuned`; o padrão é `original`.

## Imagens e anotações

`data/ocr/manifest.csv` regista origem, autoria, licença, divisão e transcrições de referência verificadas visualmente. As referências não são saídas do OCR. O conjunto contém apenas duas imagens-fonte de treino/validação e duas imagens de teste; é um piloto minúsculo, sem treino do detetor, inadequado para alegar generalização ou uso em produção.

## Limitações

O OCR pode omitir texto manuscrito, estilizado, pequeno ou com pouco contraste e pode devolver erros com confiança elevada. A avaliação usa apenas duas imagens; o teste não mostrou qualquer alteração das métricas após o fine-tuning. O modelo afinado é uma alternativa experimental, com a inferência limitada a um recorte por lote. Confirma sempre informação importante.
# ocr-training-image
