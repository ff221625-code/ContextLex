let selectedModule = "sense";

function selectModule(module, button) {

    selectedModule = module;

    document.querySelectorAll(".module-btn")
        .forEach(b => b.classList.remove("active"));

    button.classList.add("active");

    document.getElementById("target-section").style.display =
        module === "sense" ? "block" : "none";

    document.getElementById("result").innerHTML = `
        <h2>Result</h2>
        <p class="placeholder">
            Enter a sentence and click Analyze.
        </p>`;
}


async function analyze() {

    const sentence =
        document.getElementById("sentence").value.trim();

    const target =
        document.getElementById("target").value.trim();

    const result =
        document.getElementById("result");

    if (!sentence) {
        result.innerHTML =
            "<h2>Error</h2><p>Please enter a sentence.</p>";
        return;
    }

    result.innerHTML =
        "<h2>Analyzing...</h2>";

    try {

        const response = await fetch("/analyze", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                sentence: sentence,
                target: target,
                module: selectedModule
            })
        });

        const data = await response.json();

        if (!data.success) {
            result.innerHTML =
                `<h2>Error</h2><p>${data.message}</p>`;
            return;
        }

        if (selectedModule === "sense") {

            result.innerHTML = `
                <h2>🧠 Word Sense Analysis</h2>
                <div class="result-item">
                    <b>Target Word:</b> ${data.word}
                </div>
                <div class="result-item">
                    <b>Meaning:</b> ${data.meaning}
                </div>
                <div class="result-item">
                    <b>Sense:</b> ${data.sense}
                </div>
                <div class="result-item">
                    <b>Example:</b> ${data.example}
                </div>`;
        }

        else if (selectedModule === "semantic") {

            result.innerHTML = `
                <h2>🔍 Semantic Role Analysis</h2>
                <div class="result-item">
                    <b>Subject / Agent:</b>
                    ${data.subject}
                </div>
                <div class="result-item">
                    <b>Action / Predicate:</b>
                    ${data.action}
                </div>
                <div class="result-item">
                    <b>Object / Argument:</b>
                    ${data.object}
                </div>`;
        }

        else {

            let words = data.predictions.map(
                word => `<span class="prediction">${word}</span>`
            ).join(" ");

            result.innerHTML = `
                <h2>✨ Next-Word Prediction</h2>
                <div class="result-item">
                    <b>Possible Next Words:</b><br><br>
                    ${words}
                </div>`;
        }

    } catch (error) {

        console.error(error);

        result.innerHTML = `
            <h2>Connection Error</h2>
            <p>
                Unable to connect to the ContextLex server.
                Make sure app.py is running.
            </p>`;
    }
}