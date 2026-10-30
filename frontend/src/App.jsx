import { useEffect, useRef, useState } from "react";
import { extractText as requestOcr } from "./services/ocr.js";

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];
const MAX_FILE_SIZE = 10 * 1024 * 1024;

function UploadIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 16V4m0 0L7 9m5-5 5 5M5 15v4a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-4" />
    </svg>
  );
}

function CopyIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <rect x="8" y="8" width="12" height="12" rx="2" />
      <path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3" />
    </svg>
  );
}

function ImageIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <rect x="3" y="4" width="18" height="16" rx="3" />
      <circle cx="8.5" cy="9" r="1.5" />
      <path d="m21 15-5-5L5 20" />
    </svg>
  );
}

function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [dragging, setDragging] = useState(false);
  const [elapsed, setElapsed] = useState(null);
  const [model, setModel] = useState("original");
  const inputRef = useRef(null);
  const requestRef = useRef(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return undefined;
    }

    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(
    () => () => {
      requestRef.current?.abort();
    },
    [],
  );

  function chooseFile(candidate) {
    if (!candidate) return;

    setError("");
    setNotice("");
    setElapsed(null);
    setText("");
    requestRef.current?.abort();
    setLoading(false);

    if (!ACCEPTED_TYPES.includes(candidate.type)) {
      setFile(null);
      setError("Escolha uma imagem JPEG, PNG ou WEBP.");
      return;
    }
    if (candidate.size > MAX_FILE_SIZE) {
      setFile(null);
      setError("A imagem deve ter no máximo 10 MB.");
      return;
    }

    setFile(candidate);
  }

  async function extractText() {
    if (!file || loading) return;

    const controller = new AbortController();
    requestRef.current = controller;
    setLoading(true);
    setError("");
    setNotice("");
    setText("");
    setElapsed(null);

    try {
      const payload = await requestOcr(file, model, controller.signal);

      setText(payload.text ?? "");
      setElapsed(payload.processing_time_ms);
      if (!payload.text?.trim()) {
        setNotice("Não foi encontrado texto nesta imagem. Tenta uma imagem mais nítida.");
      }
    } catch (requestError) {
      if (requestError.name !== "AbortError") {
        setError(
          requestError instanceof TypeError
            ? "Não foi possível ligar ao serviço OCR. Confirma se o backend está em execução."
            : requestError.message || "Não foi possível processar a imagem.",
        );
      }
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = null;
        setLoading(false);
      }
    }
  }

  async function copyText() {
    try {
      await navigator.clipboard.writeText(text);
      setNotice("Texto copiado para a área de transferência.");
      setError("");
    } catch {
      setError("Não foi possível copiar. Seleciona o texto e copia manualmente.");
      setNotice("");
    }
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragging(false);
    chooseFile(event.dataTransfer.files?.[0]);
  }

  function reset() {
    requestRef.current?.abort();
    setFile(null);
    setText("");
    setError("");
    setNotice("");
    setElapsed(null);
    setLoading(false);
    if (inputRef.current) inputRef.current.value = "";
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="Imagem para texto — início">
          <span className="brand-mark"><ImageIcon /></span>
          <span>imagem<span className="brand-light">para</span>texto</span>
        </a>
        <span className="local-badge"><span /> OCR local</span>
      </header>

      <section className="intro" aria-labelledby="page-title">
        <p className="eyebrow">FERRAMENTA OCR</p>
        <h1 id="page-title">As palavras, sem esforço.</h1>
        <p className="intro-copy">
          Carregue uma imagem e transforme o texto visível em conteúdo que pode editar e copiar.
        </p>
      </section>

      <section className="workspace" aria-label="Leitor de imagem e texto reconhecido">
        <article className="panel image-panel">
          <div className="panel-heading">
            <div>
              <span className="step-label">PASSO 01</span>
              <h2>Imagem de origem</h2>
            </div>
            {file && (
              <button className="text-button" type="button" onClick={reset}>
                Remover
              </button>
            )}
          </div>

          <input
            ref={inputRef}
            className="visually-hidden"
            type="file"
            accept="image/jpeg,image/png,image/webp"
            aria-label="Escolher imagem para OCR"
            onChange={(event) => {
              chooseFile(event.target.files?.[0]);
              event.target.value = "";
            }}
          />

          {previewUrl ? (
            <div className="preview-frame">
              <img src={previewUrl} alt={`Pré-visualização de ${file.name}`} />
              <div className="image-caption">
                <span className="file-indicator" />
                <span className="file-name" title={file.name}>{file.name}</span>
                <span className="file-size">{(file.size / (1024 * 1024)).toFixed(2)} MB</span>
              </div>
            </div>
          ) : (
            <button
              type="button"
              className={`dropzone${dragging ? " is-dragging" : ""}`}
              onClick={() => inputRef.current?.click()}
              onDragOver={(event) => {
                event.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              aria-describedby="upload-hint"
            >
              <span className="upload-icon"><UploadIcon /></span>
              <span className="drop-title">Arraste uma imagem para aqui</span>
              <span className="drop-subtitle">ou <span className="browse-link">escolha um ficheiro</span></span>
              <span id="upload-hint" className="file-hint">JPEG, PNG ou WEBP · até 10 MB</span>
            </button>
          )}

          <div className="image-footer">
            <span className="privacy-icon" aria-hidden="true">⌁</span>
            <span>Processamento local; as imagens não ficam guardadas no servidor.</span>
          </div>
        </article>

        <article className="panel text-panel">
          <div className="panel-heading">
            <div>
              <span className="step-label">PASSO 02</span>
              <h2>Texto reconhecido</h2>
            </div>
            <button
              className="copy-button"
              type="button"
              onClick={copyText}
              disabled={!text.trim() || loading}
              aria-label="Copiar texto reconhecido"
            >
              <CopyIcon />
              <span>Copiar</span>
            </button>
          </div>

          <div className="editor-wrap">
            {loading && (
              <div className="loading-overlay" role="status" aria-live="polite">
                <span className="spinner" />
                <span>A ler a imagem…</span>
              </div>
            )}
            <label className="visually-hidden" htmlFor="recognized-text">Texto reconhecido, editável</label>
            <textarea
              id="recognized-text"
              className="text-editor"
              value={text}
              onChange={(event) => setText(event.target.value)}
              placeholder="O texto extraído aparecerá aqui. Podes editá-lo depois do reconhecimento."
              spellCheck="false"
              aria-describedby="editor-hint"
              readOnly={loading}
            />
            {!text && !loading && (
              <span className="editor-placeholder-icon" aria-hidden="true">Aa</span>
            )}
          </div>

          <div className="editor-footer">
            <span id="editor-hint">
              {text ? `${text.length} caracteres · podes editar o resultado` : "O resultado será editável"}
            </span>
            {elapsed !== null && <span>{Math.round(elapsed)} ms</span>}
          </div>
        </article>
      </section>

      {(error || notice) && (
        <div className={`feedback${error ? " feedback-error" : " feedback-success"}`} role={error ? "alert" : "status"} aria-live="polite">
          <span aria-hidden="true">{error ? "!" : "✓"}</span>
          <span>{error || notice}</span>
        </div>
      )}

      <div className="action-row">
        <label className="model-picker">
          <span className="engine-dot" />
          <span className="visually-hidden">Modelo OCR</span>
          <select
            value={model}
            onChange={(event) => setModel(event.target.value)}
            aria-label="Modelo OCR"
          >
            <option value="original">PP-OCRv4 original</option>
            <option value="finetuned">Fine-tuned · piloto</option>
          </select>
        </label>
        <button
          className="primary-button"
          type="button"
          onClick={extractText}
          disabled={!file || loading}
        >
          {loading ? <><span className="button-spinner" /> A processar…</> : <>Extrair texto <span aria-hidden="true">→</span></>}
        </button>
      </div>

      <footer className="page-footer">
        <span>Resultados automáticos podem conter erros. Confirme dados importantes.</span>
        <span>OCR · PRIVACIDADE PRIMEIRO</span>
      </footer>
    </main>
  );
}

export default App;
