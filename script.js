const form = document.getElementById("predictionForm");

const predictButton = document.getElementById("predictButton");
const loading = document.getElementById("loading");

const resultBox = document.getElementById("result");
const resultTitle = document.getElementById("resultTitle");
const probability = document.getElementById("probability");


form.addEventListener("submit", async function (event) {

    event.preventDefault();

    predictButton.disabled = true;
    loading.classList.remove("hidden");
    resultBox.classList.add("hidden");


    const data = {

        age: Number(document.getElementById("age").value),

        sex: Number(document.getElementById("sex").value),

        cp: Number(document.getElementById("cp").value),

        trestbps: Number(
            document.getElementById("trestbps").value
        ),

        chol: Number(
            document.getElementById("chol").value
        ),

        fbs: Number(
            document.getElementById("fbs").value
        ),

        restecg: Number(
            document.getElementById("restecg").value
        ),

        thalach: Number(
            document.getElementById("thalach").value
        ),

        exang: Number(
            document.getElementById("exang").value
        ),

        oldpeak: Number(
            document.getElementById("oldpeak").value
        ),

        slope: Number(
            document.getElementById("slope").value
        ),

        ca: Number(
            document.getElementById("ca").value
        ),

        thal: Number(
            document.getElementById("thal").value
        )
    };


    try {

        const response = await fetch("/api/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(data)

        });


        const result = await response.json();


        if (!response.ok) {
            throw new Error(
                result.error || "Prediction failed"
            );
        }


        resultTitle.textContent = result.result;

        probability.textContent =
            result.probability + "%";


        resultBox.classList.remove("hidden");


    } catch (error) {

        resultTitle.textContent = "Error";

        probability.textContent = error.message;

        resultBox.classList.remove("hidden");

    }


    loading.classList.add("hidden");

    predictButton.disabled = false;

});
