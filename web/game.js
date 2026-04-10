const WINS = [
  [0,1,2],[3,4,5],[6,7,8],
  [0,3,6],[1,4,7],[2,5,8],
  [0,4,8],[2,4,6]
];

let board, current, over;
let score = { X: 0, O: 0, D: 0 };
let session = null;
let lastProbs = null;
let lastLogits = null;

const cells   = document.querySelectorAll('.cell');
const status  = document.getElementById('status');
const resetBtn = document.getElementById('reset');
const makeMoveBtn = document.getElementById('make-move');
const sx = document.getElementById('sx');
const so = document.getElementById('so');
const sd = document.getElementById('sd');
const selectModel = document.getElementById('ai-model');
const tempSlider = document.getElementById('temperature');
const tempVal = document.getElementById('temp-val');
const inferenceTimeEl = document.getElementById('inference-time');

function init() {
  board   = Array(9).fill(null);
  current = 'X';
  over    = false;
  cells.forEach(c => {
    c.textContent = '';
    c.className = 'cell';
    c.style.backgroundColor = '';
  });
  setStatus(`${current}'s turn`);
  document.getElementById('value-eval').textContent = '-';
  updateHeatmap();
}

function setStatus(msg, type = '') {
  status.textContent = msg;
  status.className = type;
}

function checkWin(player) {
  return WINS.find(line => line.every(i => board[i] === player)) || null;
}

function handleClick(e) {
  const cell = e.currentTarget;
  const i = +cell.dataset.i;
  if (over || board[i]) return;

  board[i] = current;
  cell.textContent = current;
  cell.style.backgroundColor = '';
  cell.classList.add(current.toLowerCase(), 'taken');

  const winLine = checkWin(current);
  if (winLine) {
    winLine.forEach(idx => cells[idx].classList.add('win'));
    setStatus(`${current} wins!`, 'winner');
    score[current]++;
    updateScore();
    over = true;
    updateHeatmap();
    return;
  }

  if (board.every(Boolean)) {
    setStatus("It's a draw!", 'draw');
    score.D++;
    updateScore();
    over = true;
    updateHeatmap();
    return;
  }

  current = current === 'X' ? 'O' : 'X';
  setStatus(`${current}'s turn`);
  updateHeatmap();
}

function updateScore() {
  sx.textContent = score.X;
  so.textContent = score.O;
  sd.textContent = score.D;
}

cells.forEach(c => c.addEventListener('click', handleClick));
resetBtn.addEventListener('click', init);

makeMoveBtn.addEventListener('click', () => {
  if (over || !lastProbs) return;
  
  const rand = Math.random();
  let cumulative = 0;
  let moveIdx = -1;
  
  for (let i = 0; i < 9; i++) {
    if (board[i] === null) {
      cumulative += lastProbs[i];
      if (rand <= cumulative) {
        moveIdx = i;
        break;
      }
    }
  }
  
  // Precision fallback
  if (moveIdx === -1) {
    moveIdx = board.findIndex(v => v === null);
  }
  
  if (moveIdx !== -1) {
    // Manually push fake click event
    handleClick({ currentTarget: cells[moveIdx] });
  }
});

async function loadAvailableModels() {
  try {
    const response = await fetch('../model/models.json');
    if (!response.ok) throw new Error("models.json missing");
    
    const files = await response.json();
    
    selectModel.innerHTML = '<option value="">None</option>';
    
    if (files.length > 0) {
      files.forEach(fileName => {
        const option = document.createElement('option');
        option.value = `../model/${fileName}`;
        option.textContent = fileName.replace('.onnx', '');
        selectModel.appendChild(option);
      });
    } else {
      throw new Error("No ONNX files tracking in models.json.");
    }
  } catch (error) {
    console.warn("Could not automatically load models. Fallback to default list.", error);
    const defaults = ['mlp32', 'mlp32_deepq', 'mlp64', 'mlp64_deepq'];
    selectModel.innerHTML = '<option value="">None</option>';
    defaults.forEach(file => {
      const option = document.createElement('option');
      option.value = `../model/${file}.onnx`;
      option.textContent = file;
      selectModel.appendChild(option);
    });
  }
  if (selectModel.options.length > 0) {
    await loadModel(selectModel.options[0].value);
  }
}

async function loadModel(path) {
  if (!path) {
    session = null;
    updateHeatmap();
    return;
  }
  try {
    session = await ort.InferenceSession.create(path);
    updateHeatmap();
  } catch (error) {
    console.error("Failed to load model:", error);
  }
}

function softmax(logits, temp) {
  const maxLogit = Math.max(...logits);
  if (temp === 0.0) {
    const maxCount = logits.filter(l => l === maxLogit).length;
    return logits.map(l => (l === maxLogit ? 1.0 / maxCount : 0.0));
  }
  const exps = logits.map(x => Math.exp((x - maxLogit) / temp));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sumExps);
}

async function updateHeatmap() {
  makeMoveBtn.disabled = true;

  if (over || !session) {
    inferenceTimeEl.textContent = "";
    document.getElementById('value-eval').textContent = "-";
    cells.forEach((c, i) => {
      c.style.backgroundColor = '';
      if (board && board[i] === null) c.innerHTML = '';
    });
    return;
  }
  
  inferenceTimeEl.textContent = "Inferencing...";
  
  const inputData = new Float32Array(9);
  for (let i = 0; i < 9; i++) {
    if (board[i] === current) inputData[i] = 1.0;
    else if (board[i] !== null) inputData[i] = -1.0;
    else inputData[i] = 0.0;
  }
  
  try {
    const start = performance.now();
    const tensor = new ort.Tensor('float32', inputData, [1, 9]);
    const results = await session.run({ "input": tensor });
    const end = performance.now();
    
    inferenceTimeEl.textContent = `Inference time: ${(end - start).toFixed(1)} ms`;

    const logits = Array.from(results["policy"].data);
    let value = Array.from(results["value"].data)[0];
    
    // Convert value to absolute perspective (X's POV)
    if (current === 'O') {
      value = -value;
    }
    
    // Display value
    document.getElementById('value-eval').textContent = (value > 0 ? '+' : '') + value.toFixed(4);
    
    // Mask out taken cells
    for (let i = 0; i < 9; i++) {
      if (board[i] !== null) logits[i] = -Infinity;
    }
    
    lastLogits = logits;
    renderHeatmap();

    makeMoveBtn.disabled = false;
  } catch (error) {
    console.error("Error in ONNX inference:", error);
    inferenceTimeEl.textContent = "Inference failed";
  }
}

function renderHeatmap() {
  if (over || !lastLogits) return;
  
  const temp = parseFloat(tempSlider.value);
  const probs = softmax(lastLogits, temp);
  lastProbs = [...probs];
  
  cells.forEach((cell, i) => {
    if (board[i] !== null) {
      cell.style.backgroundColor = '';
    } else {
      const prob = Math.round(probs[i] * 100);
      const intensity = (prob * 0.3).toFixed(1);
      cell.style.backgroundColor = `color-mix(in oklch, var(--color-${current.toLowerCase()}) ${intensity}%, #ffffff)`;
      cell.innerHTML = `<span style="font-size: 1.2rem; color: color-mix(in srgb, var(--color-accent), black 40%); font-weight: 500;">${prob}%</span>`;
    }
  });
}

selectModel.addEventListener('change', async (e) => {
  await loadModel(e.target.value);
});

function updateSliderUI() {
  const min = parseFloat(tempSlider.min);
  const max = parseFloat(tempSlider.max);
  const val = parseFloat(tempSlider.value);
  const percent = ((val - min) / (max - min)) * 100;
  
  // Use CSS property so we can keep code clean
  const accent = getComputedStyle(document.documentElement).getPropertyValue('--color-accent').trim();
  tempSlider.style.background = `linear-gradient(to right, ${accent} 0%, ${accent} ${percent}%, #94a3b8 ${percent}%, #94a3b8 100%)`;
}

tempSlider.addEventListener('input', (e) => {
  tempVal.textContent = parseFloat(e.target.value).toFixed(1);
  updateSliderUI();
  renderHeatmap();
});

updateSliderUI();
loadAvailableModels().then(() => init());
