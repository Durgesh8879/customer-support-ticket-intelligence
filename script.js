const API_URL = 'https://customer-support-ticket-intelligence.onrender.com/predict';
const form = document.querySelector("#ticket-form");
const messageInput = document.querySelector("#ticket-message");
const characterCount = document.querySelector("#character-count");
const analyzeButton = document.querySelector("#analyze-button");
const clearButton = document.querySelector("#clear-button");
const feedback = document.querySelector("#form-feedback");
const resultBadge = document.querySelector("#result-badge");
const emptyResult = document.querySelector("#empty-result");
const loadingResult = document.querySelector("#loading-result");
const predictionResult = document.querySelector("#prediction-result");
const errorResult = document.querySelector("#error-result");
const predictedIntent = document.querySelector("#predicted-intent");
const intentIdentifier = document.querySelector("#intent-identifier");
const returnedMessage = document.querySelector("#returned-message");
const errorMessage = document.querySelector("#error-message");

const resultPanels = [emptyResult, loadingResult, predictionResult, errorResult];
let activeRequestController = null;

function showPanel(panel) {
  for (const resultPanel of resultPanels) {
    resultPanel.hidden = resultPanel !== panel;
  }
}

function setBadge(label, state = "") {
  resultBadge.textContent = label;
  resultBadge.className = `result-badge${state ? ` is-${state}` : ""}`;
}

function updateCharacterCount() {
  const count = messageInput.value.length;
  characterCount.textContent = `${count} ${count === 1 ? "character" : "characters"}`;
}

function formatIntent(intent) {
  return intent
    .split("_")
    .filter(Boolean)
    .map((word) => `${word.charAt(0).toUpperCase()}${word.slice(1).toLowerCase()}`)
    .join(" ");
}

function setLoading(isLoading) {
  analyzeButton.disabled = isLoading;
  analyzeButton.querySelector(".button-label").textContent = isLoading
    ? "Analyzing..."
    : "Analyze Ticket";
  analyzeButton.setAttribute("aria-busy", String(isLoading));
}

function showError(message) {
  errorMessage.textContent = message;
  feedback.textContent = "";
  showPanel(errorResult);
  setBadge("ERROR", "error");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (analyzeButton.disabled) return;

  const message = messageInput.value.trim();
  if (!message) {
    feedback.textContent = "Enter a customer message before analyzing.";
    messageInput.focus();
    return;
  }

  feedback.textContent = "";
  setLoading(true);
  showPanel(loadingResult);
  setBadge("ANALYZING", "loading");

  const requestController = new AbortController();
  activeRequestController = requestController;

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message }),
      signal: requestController.signal,
    });

    if (!response.ok) {
      throw new Error(`The API returned HTTP ${response.status}. Please try again.`);
    }

    let data;
    try {
      data = await response.json();
    } catch {
      throw new Error("The API response was not valid JSON.");
    }

    if (
      !data ||
      typeof data.predicted_intent !== "string" ||
      !data.predicted_intent.trim()
    ) {
      throw new Error("The API response is missing a valid predicted_intent.");
    }

    predictedIntent.textContent = formatIntent(data.predicted_intent);
    intentIdentifier.textContent = data.predicted_intent;
    returnedMessage.textContent =
      typeof data.customer_message === "string" ? data.customer_message : message;
    showPanel(predictionResult);
    setBadge("COMPLETE", "success");
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      return;
    } else if (error instanceof TypeError) {
      showError(
        "Could not reach the API. Check that FastAPI is running at 127.0.0.1:8000 and that CORS allows this page's origin."
      );
    } else {
      showError(error instanceof Error ? error.message : "An unexpected error occurred.");
    }
  } finally {
    if (activeRequestController === requestController) {
      activeRequestController = null;
      setLoading(false);
    }
  }
});

messageInput.addEventListener("input", () => {
  updateCharacterCount();
  if (feedback.textContent) feedback.textContent = "";
});

messageInput.addEventListener("keydown", (event) => {
  if (event.ctrlKey && event.key === "Enter") {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll(".example-chip").forEach((button) => {
  button.addEventListener("click", () => {
    messageInput.value = button.dataset.example;
    updateCharacterCount();
    feedback.textContent = "";
    messageInput.focus();
  });
});

clearButton.addEventListener("click", () => {
  activeRequestController?.abort();
  activeRequestController = null;
  setLoading(false);
  messageInput.value = "";
  updateCharacterCount();
  feedback.textContent = "";
  showPanel(emptyResult);
  setBadge("READY");
  messageInput.focus();
});

updateCharacterCount();
