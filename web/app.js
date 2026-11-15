const form = document.querySelector('#ask-form');
const question = document.querySelector('#question');
const button = document.querySelector('#ask-button');
const status = document.querySelector('#status');
const result = document.querySelector('#result');
const evidenceCard = document.querySelector('#evidence-card');
const evaluateButton = document.querySelector('#evaluate-button');
const fullEvaluateButton = document.querySelector('#full-evaluate-button');
const uploadForm = document.querySelector('#upload-form');
const uploadButton = document.querySelector('#upload-button');
const documentsInput = document.querySelector('#documents');

documentsInput.addEventListener('change', () => {
  const names = Array.from(documentsInput.files).map(file => file.name);
  document.querySelector('#selected-files').textContent = names.length
    ? `${names.length} file(s) selected: ${names.join(', ')}`
    : 'No files selected.';
});

evaluateButton.addEventListener('click', () => runEvaluation(false));
fullEvaluateButton.addEventListener('click', () => runEvaluation(true));

uploadForm.addEventListener('submit', async event => {
  event.preventDefault();
  const files = documentsInput.files;
  if (!files.length) return;
  uploadButton.disabled = true;
  status.textContent = 'Uploading and indexing...';
  status.className = 'status';
  try {
    const response = await fetch('/api/upload', {
      method: 'POST',
      body: new FormData(uploadForm)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Upload failed');
    status.textContent = `${data.files.length} document(s) indexed`;
    documentsInput.value = '';
    document.querySelector('#selected-files').textContent = 'No files selected.';
  } catch (error) {
    status.textContent = error.message;
    status.className = 'status error';
  } finally {
    uploadButton.disabled = false;
  }
});

async function runEvaluation(full) {
  const activeButton = full ? fullEvaluateButton : evaluateButton;
  activeButton.disabled = true;
  activeButton.textContent = full ? 'Checking answers...' : 'Checking retrieval...';
  try {
    const response = await fetch(full ? '/api/evaluate/full' : '/api/evaluate', {
      method: full ? 'POST' : 'GET'
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Evaluation failed');
    if (full) renderFullEvaluation(data);
    else renderEvaluation(data);
  } catch (error) {
    status.textContent = error.message;
    status.className = 'status error';
  } finally {
    activeButton.disabled = false;
    activeButton.textContent = full ? 'Full answer check' : 'Run retrieval report';
  }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const value = question.value.trim();
  if (!value) return;
  button.disabled = true;
  status.textContent = 'Searching documents...';
  status.className = 'status';
  try {
    const response = await fetch('/api/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: value})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Request failed');
    render(data);
    status.textContent = 'Complete';
  } catch (error) {
    status.textContent = error.message;
    status.className = 'status error';
  } finally {
    button.disabled = false;
  }
});

function render(data) {
  result.classList.remove('hidden');
  evidenceCard.classList.remove('hidden');
  document.querySelector('#answer').textContent = data.answer;
  document.querySelector('#answer-time').textContent = `${data.evidence.length} chunks used`;
  document.querySelector('#evidence-count').textContent = 'Inspect the retrieved context';
  document.querySelector('#references').innerHTML = data.sources.map(source =>
    `<div class="reference"><strong>[${source.id}]</strong> ${escapeHtml(source.source)} - chunk ${source.chunk_index}</div>`
  ).join('');
  document.querySelector('#evidence').innerHTML = data.evidence.map((item, i) =>
    `<details class="evidence" ${i === 0 ? 'open' : ''}><summary>[${item.id}] ${escapeHtml(item.source)}</summary><div class="meta">${scoreSummary(item)} · Chunk: ${item.chunk_index}</div><p>${escapeHtml(item.text)}</p></details>`
  ).join('');
}

function scoreSummary(item) {
  const distance = item.distance == null ? '' : ` · Distance: ${Number(item.distance).toFixed(4)}`;
  return `Final: ${formatScore(item.score)} · Semantic: ${formatScore(item.semantic_score)} · BM25: ${formatScore(item.keyword_score)}${distance}`;
}

function formatScore(value) {
  return Number(value || 0).toFixed(3);
}

function renderEvaluation(data) {
  document.querySelector('#evaluation-card').classList.remove('hidden');
  document.querySelector('#evaluation-title').textContent = 'Retrieval quality report';
  document.querySelector('#evaluation-summary').textContent = `${data.passed}/${data.total} evidence checks passed`;
  document.querySelector('#evaluation-quality').innerHTML =
    `<strong>${escapeHtml(data.quality_label || 'Retrieval quality')}</strong>` +
    `<p>${escapeHtml(data.quality_explanation || '')}</p>`;
  document.querySelector('#evaluation-metrics').innerHTML = [
    metricCard(formatPercentage(data.pass_percentage), 'Test pass rate', `${data.passed}/${data.total} cases passed`),
    metricCard(formatPercentage(data.recall_percentage), `Recall @${data.k}`, 'Relevant evidence found'),
    metricCard(formatPercentage(data.precision_percentage), `Precision @${data.k}`, 'Retrieved evidence that is relevant'),
    metricCard(formatPercentage(data.f1_percentage), `F1 @${data.k}`, 'Balance of Recall and Precision'),
    metricCard(formatScore(data.mean_reciprocal_rank), 'MRR', 'How early relevant evidence appears'),
    metricCard(formatPercentage(data.no_answer_percentage), 'Abstention accuracy', `${data.no_answer_passed}/${data.no_answer_total} unsupported cases`),
  ].join('');
  document.querySelector('#evaluation-explanation').textContent =
    `Why these numbers matter: Recall shows whether important facts are missed; Precision shows whether irrelevant chunks add noise; F1 combines both; MRR shows whether useful evidence appears near the top; the pass rate shows overall test behavior. ` +
    `The report uses ${data.metric_cases} answerable retrieval cases and ${data.total} total test cases.`;
  document.querySelector('#evaluation-results').innerHTML = data.results.map(item => evaluationRow(item)).join('');
  addReviewHandlers();
}

function metricCard(value, label, detail) {
  return `<div class="metric"><strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span><small>${escapeHtml(detail)}</small></div>`;
}

function renderFullEvaluation(data) {
  document.querySelector('#evaluation-card').classList.remove('hidden');
  document.querySelector('#evaluation-title').textContent = 'Full answer check';
  document.querySelector('#evaluation-summary').textContent = `${data.passed}/${data.total} answers passed`;
  document.querySelector('#evaluation-quality').innerHTML = '';
  document.querySelector('#evaluation-explanation').textContent = 'This check evaluates generated answers, expected terms, and citation validity. It is slower because it calls the language model.';
  document.querySelector('#evaluation-metrics').innerHTML =
    `<div class="metric"><strong>${data.passed}/${data.total}</strong><span>Answers passed</span></div>` +
    `<div class="metric"><strong>${data.needs_review}</strong><span>Needs review</span></div>`;
  document.querySelector('#evaluation-results').innerHTML = data.results.map(item =>
    `<details class="evaluation-row"><summary><span>${escapeHtml(item.question)}</span><span class="${item.passed ? 'pass' : 'fail'}">${item.passed ? 'PASS' : 'CHECK'}</span></summary><div class="meta"><strong>Expected:</strong> ${escapeHtml(item.expected_answer)}</div><div class="evaluation-evidence"><strong>Generated:</strong> ${escapeHtml(item.answer)}</div><div class="meta ${item.valid_citations ? 'pass' : 'fail'}"><strong>Citations:</strong> ${escapeHtml(item.citations.join(', ') || 'none')} - ${item.valid_citations ? 'valid' : 'missing or invalid'}</div></details>`
  ).join('');
}

function evaluationRow(item) {
  const key = encodeURIComponent(item.question);
  const saved = localStorage.getItem(`rag-review-${key}`);
  const details = item.missing_terms.length ? item.missing_terms : item.found_terms;
  const metrics = item.precision_percentage == null
    ? 'Not applicable for this question'
    : `Precision ${formatPercentage(item.precision_percentage)} · Recall ${formatPercentage(item.recall_percentage)}`;
  return `<details class="evaluation-row"><summary><span>${escapeHtml(item.question)}</span><span class="${item.passed ? 'pass' : 'fail'}">${item.passed ? 'CORRECT' : 'CHECK'}</span></summary><div class="meta"><strong>Expected answer:</strong> ${escapeHtml(item.expected_answer)}</div><div class="meta"><strong>Retrieved:</strong> ${escapeHtml(item.top_source || 'none')} - chunk ${item.top_chunk ?? '-'}</div><div class="meta"><strong>Retrieval metrics:</strong> ${metrics}</div><div class="evaluation-evidence">${escapeHtml(item.retrieved_text || 'No chunk was retrieved.')}</div><div class="meta ${item.missing_terms.length ? 'fail' : 'pass'}"><strong>${item.missing_terms.length ? 'Missing:' : 'Found:'}</strong> ${escapeHtml(details.join(', ') || 'none')}</div><div class="review-actions"><span>Manual review:</span><button data-review="correct" data-question="${key}" class="review-button ${saved === 'correct' ? 'selected' : ''}">Correct</button><button data-review="needs-review" data-question="${key}" class="review-button ${saved === 'needs-review' ? 'selected' : ''}">Needs review</button></div></details>`;
}

function formatPercentage(value) {
  return value == null ? '—' : `${Number(value).toFixed(1)}%`;
}

function addReviewHandlers() {
  document.querySelectorAll('.review-button').forEach(button => button.addEventListener('click', event => {
    event.preventDefault();
    const key = button.dataset.question;
    localStorage.setItem(`rag-review-${key}`, button.dataset.review);
    button.parentElement.querySelectorAll('.review-button').forEach(item => item.classList.remove('selected'));
    button.classList.add('selected');
  }));
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, char => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
  }[char]));
}
