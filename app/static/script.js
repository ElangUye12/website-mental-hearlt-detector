// app/static/script.js

const DIAGNOSIS_MAPPING = {
  "No Depression": {
    label: "MINIMAL",
    confidence_text: "Risiko Sangat Rendah",
    color: "#28a745", // Green
    confidence_level: "Medium",
  },
  Mild: {
    label: "RINGAN",
    confidence_text: "Risiko Ringan",
    color: "#ffc107", // Yellow
    confidence_level: "Medium",
  },
  Moderate: {
    label: "SEDANG",
    confidence_text: "Risiko Sedang",
    color: "#fd7e14", // Orange
    confidence_level: "Medium",
  },
  Severe: {
    label: "BERAT",
    confidence_text: "Risiko Berat",
    color: "#dc3545", // Red
    confidence_level: "Medium",
  },
};

let currentStep = 1;
const totalSteps = 14;
const steps = document.querySelectorAll(".step");
const prevBtn = document.getElementById("prevBtn");
const nextBtn = document.getElementById("nextBtn");
const submitBtn = document.getElementById("submitBtn");
const currentStepSpan = document.getElementById("currentStep");
const progressBarFill = document.getElementById("progressBarFill");
const answeredCountSpan = document.getElementById("answeredCount");
const form = document.getElementById("detectionForm");

function updateProgress() {
  const answered = Array.from(steps).filter((step) => {
    const radios = step.querySelectorAll('input[type="radio"]');
    // Cek radio button
    if (radios.length > 0) {
      return Array.from(radios).some((r) => r.checked);
    }
    // Cek input number (Age)
    const numberInput = step.querySelector('input[type="number"]');
    if (numberInput) {
      return numberInput.value.trim() !== "";
    }
    return false;
  }).length;

  const progress = (answered / totalSteps) * 100;
  progressBarFill.style.width = `${progress}%`;
  answeredCountSpan.textContent = answered;
}

function showStep(stepIndex) {
  steps.forEach((step, i) => {
    step.classList.toggle("active", i === stepIndex - 1);
  });
  currentStepSpan.textContent = stepIndex;

  prevBtn.style.display = stepIndex === 1 ? "none" : "inline-block";

  if (stepIndex === totalSteps) {
    nextBtn.style.display = "none";
    submitBtn.style.display = "inline-block";
  } else {
    nextBtn.style.display = "inline-block";
    submitBtn.style.display = "none";
  }

  steps[stepIndex - 1].scrollIntoView({ behavior: "smooth", block: "start" });
}

function validateCurrentStep() {
  const step = steps[currentStep - 1];
  const radios = step.querySelectorAll('input[type="radio"]');
  const numberInput = step.querySelector('input[type="number"]');

  if (radios.length > 0) {
    return Array.from(radios).some((r) => r.checked);
  }
  if (numberInput) {
    const val = numberInput.value.trim();
    // Validasi sederhana untuk usia
    return val !== "" && !isNaN(val) && parseFloat(val) >= 0;
  }
  return true;
}

form.addEventListener("change", updateProgress);
form.addEventListener("input", updateProgress); // Tambah event untuk input number

prevBtn.addEventListener("click", () => {
  if (currentStep > 1) {
    currentStep--;
    showStep(currentStep);
  }
});

nextBtn.addEventListener("click", () => {
  if (validateCurrentStep()) {
    currentStep++;
    showStep(currentStep);
  } else {
    alert("Harap jawab pertanyaan ini sebelum melanjutkan.");
  }
});

document.addEventListener("DOMContentLoaded", () => {
  showStep(currentStep);
  updateProgress();
});

// Fungsi untuk memperbarui tampilan probabilitas di result card (Sesuai Screenshot)
function updateProbList(elementId, probabilities, diagnosisMap) {
  const list = document.getElementById(elementId);
  list.innerHTML = "";

  // Urutan label yang diinginkan: Mild, Moderate, Severe, No Depression
  const order = ["Mild", "Moderate", "Severe", "No Depression"];
  const probMap = new Map(Object.entries(probabilities));

  order.forEach((diagKey) => {
    const probValue = probMap.get(diagKey) || 0.0;
    // Hanya 1 digit di belakang koma, sesuai format tampilan hasil
    const probPercent = probValue.toFixed(1);

    const map = diagnosisMap[diagKey] || diagnosisMap["No Depression"];
    const labelText = map.label;
    const colorClass = diagKey.toLowerCase().replace(/\s/g, "-") + "-bar";

    const li = document.createElement("li");
    li.innerHTML = `
            ${labelText} 
            <div class="prob-bar-wrapper">
                <div class="prob-bar ${colorClass}" style="width: ${probPercent}%"></div>
                <span>${probPercent}%</span>
            </div>
        `;
    list.appendChild(li);
  });
}

// Event submit form
form.addEventListener("submit", function (e) {
  e.preventDefault();
  if (!validateCurrentStep()) return;

  document.getElementById("loadingOverlay").style.display = "flex";

  const formData = new FormData(e.target);
  const data = {};

  // Proses semua 14 data dari form
  formData.forEach((val, key) => {
    // PHQ-9 & Age: Nilai dikirim sebagai string/numerik.
    if (key.startsWith("phq") || key === "Age") {
      data[key] = val;
    } else {
      // Gender, family_history, treatment, work_interfere dikirim sebagai string
      data[key] = val;
    }
  });

  fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  })
    .then((res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    })
    .then((result) => {
      const svmMap =
        DIAGNOSIS_MAPPING[result.svm_diagnosis] ||
        DIAGNOSIS_MAPPING["No Depression"];
      const nbMap =
        DIAGNOSIS_MAPPING[result.nb_diagnosis] ||
        DIAGNOSIS_MAPPING["No Depression"];

      // --- Update SVM Card ---
      document.getElementById("svm_diagnosis_label").textContent = svmMap.label;
      document.getElementById("svm_diagnosis_label").style.color = svmMap.color;

      const svmConfidenceLevel = svmMap.confidence_level;
      const svmConfidenceValue = (result.svm_confidence || 0).toFixed(1);
      document.getElementById(
        "svm_confidence"
      ).textContent = `${svmConfidenceLevel} (${svmConfidenceValue}%)`;

      // Update Probabilitas SVM
      updateProbList(
        "svm_probs_list",
        result.svm_probabilities,
        DIAGNOSIS_MAPPING
      );

      // --- Update NB Card ---
      document.getElementById("nb_diagnosis_label").textContent = nbMap.label;
      document.getElementById("nb_diagnosis_label").style.color = nbMap.color;

      const nbConfidenceLevel = nbMap.confidence_level;
      const nbConfidenceValue = (result.nb_confidence || 0).toFixed(1);
      document.getElementById(
        "nb_confidence"
      ).textContent = `${nbConfidenceLevel} (${nbConfidenceValue}%)`;

      // Update Probabilitas NB
      updateProbList(
        "nb_probs_list",
        result.nb_probabilities,
        DIAGNOSIS_MAPPING
      );

      // Tampilkan hasil dan sembunyikan form/loading
      document.getElementById("result").style.display = "block";
      form.style.display = "none";
      document.getElementById("loadingOverlay").style.display = "none";
      document.getElementById("result").scrollIntoView({ behavior: "smooth" });
    })
    .catch((err) => {
      console.error("Prediction error:", err);
      alert(
        "Gagal memprediksi. Pastikan format data dan endpoint benar: " +
          err.message
      );
      document.getElementById("loadingOverlay").style.display = "none";
    });
});
