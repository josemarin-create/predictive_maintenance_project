from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "final_model.pkl"


st.set_page_config(
    page_title="Sistema de Mantenimiento Predictivo",
    layout="wide"
)

# Carga el modelo entrenado y lo mantiene en memoria para no recargarlo en cada interacción.
@st.cache_resource
def load_model_artifact():
    return joblib.load(MODEL_PATH)

# Carga el modelo entrenado y lo mantiene en memoria para no recargarlo en cada interacción.
def celsius_to_kelvin(value_celsius):
    return value_celsius + 273.15

# Clasifica la probabilidad de fallo en riesgo bajo, medio o alto.
def get_risk_level(probability):
    if probability < 0.15:
        return (
            "Riesgo bajo",
            "La máquina no muestra señales claras de fallo. Mantener seguimiento normal."
        )

    if probability < 0.30:
        return (
            "Riesgo medio",
            "La máquina presenta algunas señales de riesgo. Revisar en la próxima ronda de mantenimiento."
        )

    return (
        "Riesgo alto",
        "La máquina supera el umbral de riesgo. Priorizar revisión técnica."
    )

# Genera señales sencillas para ayudar al técnico a interpretar los valores introducidos.
def get_warning_signals(air_temp_c, process_temp_c, rotational_speed, torque, tool_wear):
    signals = []

    if air_temp_c >= 28:
        signals.append("Temperatura ambiente elevada")

    if process_temp_c >= 38:
        signals.append("Temperatura de proceso elevada")

    if rotational_speed <= 1400:
        signals.append("Velocidad de rotación baja")

    if torque >= 50:
        signals.append("Torque elevado")

    if tool_wear >= 160:
        signals.append("Desgaste de herramienta alto")

    if not signals:
        signals.append("No se observan señales operativas destacadas")

    return signals

# Construye el DataFrame con el formato exacto que espera el modelo.
def build_input_dataframe(machine_type, air_temp_c, process_temp_c, rotational_speed, torque, tool_wear):
    return pd.DataFrame(
        [{
            "Type": machine_type,
            "Air temperature [K]": celsius_to_kelvin(air_temp_c),
            "Process temperature [K]": celsius_to_kelvin(process_temp_c),
            "Rotational speed [rpm]": rotational_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear
        }]
    )

# Calcula la probabilidad de fallo y aplica el umbral de decisión.
def predict_machine(model, threshold, input_data):
    probability = model.predict_proba(input_data)[0, 1]
    prediction = int(probability >= threshold)
    return probability, prediction

# Inicializa la lista temporal donde se guardan las máquinas evaluadas durante la sesión.
def initialize_session_state():
    if "evaluated_machines" not in st.session_state:
        st.session_state.evaluated_machines = []

# Añade una máquina evaluada a la tabla temporal de seguimiento.
def add_machine_to_session(row):
    st.session_state.evaluated_machines.append(row)

# Define casos de ejemplo para probar la app sin introducir todos los datos manualmente.
def get_example_cases():
    return {
        "Máquina estable": {
            "machine_id": "MAQ-EST-001",
            "machine_type": "M",
            "air_temp_c": 25.0,
            "process_temp_c": 36.5,
            "rotational_speed": 1550,
            "torque": 38.0,
            "tool_wear": 70
        },
        "Máquina con desgaste alto": {
            "machine_id": "MAQ-DES-002",
            "machine_type": "L",
            "air_temp_c": 28.0,
            "process_temp_c": 38.0,
            "rotational_speed": 1450,
            "torque": 45.0,
            "tool_wear": 210
        },
        "Máquina con torque alto": {
            "machine_id": "MAQ-TOR-003",
            "machine_type": "M",
            "air_temp_c": 27.0,
            "process_temp_c": 37.5,
            "rotational_speed": 1350,
            "torque": 62.0,
            "tool_wear": 120
        },
        "Máquina en riesgo elevado": {
            "machine_id": "MAQ-RIE-004",
            "machine_type": "L",
            "air_temp_c": 30.0,
            "process_temp_c": 39.0,
            "rotational_speed": 1320,
            "torque": 58.0,
            "tool_wear": 190
        }
    }

# Muestra la pantalla inicial de la aplicación.
def show_home():
    st.title("Sistema de Mantenimiento Predictivo")
    st.subheader("Herramienta de apoyo para priorizar revisiones de maquinaria industrial")

    st.write(
        """
        Esta aplicación permite estimar el riesgo de fallo de una máquina a partir de variables operativas.

        Está pensada como una demo para un técnico de mantenimiento. El objetivo no es sustituir la decisión técnica,
        sino ayudar a priorizar qué máquinas conviene revisar antes.
        """
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Modelo final", "Gradient Boosting")
    col2.metric("Umbral de alerta", "0.30")
    col3.metric("Clase objetivo", "Fallo de máquina")

    st.info(
        "Esta app es una prueba de concepto basada en un dataset sintético. "
        "Para aplicarla en una empresa real habría que validarla con datos propios de maquinaria y mantenimiento."
    )

# Muestra el formulario para evaluar una máquina y obtener su riesgo de fallo.
def show_machine_evaluation(model_artifact):
    st.title("Evaluar máquina")

    model = model_artifact["model"]
    threshold = model_artifact["threshold"]

    examples = get_example_cases()
    example_name = st.selectbox(
        "Selecciona un caso de ejemplo o úsalo como base para introducir una máquina nueva",
        list(examples.keys())
    )

    example = examples[example_name]

    st.write("Puedes modificar cualquier valor antes de evaluar la máquina.")

    col1, col2 = st.columns(2)

    with col1:
        machine_id = st.text_input("ID de máquina", value=example["machine_id"])
        machine_type = st.selectbox(
            "Tipo de máquina/producto",
            ["L", "M", "H"],
            index=["L", "M", "H"].index(example["machine_type"])
        )
        air_temp_c = st.number_input(
            "Temperatura ambiente (°C)",
            min_value=0.0,
            max_value=60.0,
            value=float(example["air_temp_c"]),
            step=0.5
        )
        process_temp_c = st.number_input(
            "Temperatura de proceso (°C)",
            min_value=0.0,
            max_value=80.0,
            value=float(example["process_temp_c"]),
            step=0.5
        )

    with col2:
        rotational_speed = st.number_input(
            "Velocidad de rotación (rpm)",
            min_value=500,
            max_value=3500,
            value=int(example["rotational_speed"]),
            step=10
        )
        torque = st.number_input(
            "Torque (Nm)",
            min_value=0.0,
            max_value=100.0,
            value=float(example["torque"]),
            step=0.5
        )
        tool_wear = st.number_input(
            "Desgaste de herramienta (min)",
            min_value=0,
            max_value=300,
            value=int(example["tool_wear"]),
            step=5
        )

    input_data = build_input_dataframe(
        machine_type,
        air_temp_c,
        process_temp_c,
        rotational_speed,
        torque,
        tool_wear
    )

    if st.button("Evaluar máquina", type="primary"):
        probability, prediction = predict_machine(model, threshold, input_data)
        risk_level, recommendation = get_risk_level(probability)

        warning_signals = get_warning_signals(
            air_temp_c,
            process_temp_c,
            rotational_speed,
            torque,
            tool_wear
        )

        st.session_state["last_result"] = {
            "machine_id": machine_id,
            "machine_type": machine_type,
            "air_temp_c": air_temp_c,
            "process_temp_c": process_temp_c,
            "rotational_speed": rotational_speed,
            "torque": torque,
            "tool_wear": tool_wear,
            "failure_probability": probability,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "prediction": prediction,
            "warning_signals": warning_signals
        }

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]

        st.divider()
        st.subheader("Resultado de la evaluación")

        probability_percent = result["failure_probability"] * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Probabilidad de fallo", f"{probability_percent:.1f}%")
        col2.metric("Nivel de riesgo", result["risk_level"])
        col3.metric("Límite de alerta", f"{threshold:.2f}")


        st.write("Nivel visual de riesgo")
        st.progress(float(result["failure_probability"]))

        if result["risk_level"] == "Riesgo bajo":
            st.success(result["recommendation"])
        elif result["risk_level"] == "Riesgo medio":
            st.warning(result["recommendation"])
        else:
            st.error(result["recommendation"])

        st.subheader("Señales a revisar")
        st.caption("Estas señales son una orientación sencilla basada en los valores introducidos.")
        
        for signal in result["warning_signals"]:
            st.write(f"- {signal}")

        if st.button("Añadir a tabla de seguimiento"):
            add_machine_to_session(result)
            st.success("Máquina añadida a la tabla de seguimiento.")


# Muestra la tabla de máquinas evaluadas y permite descargarla como CSV.
def show_evaluated_machines():
    st.title("Máquinas evaluadas")

    if not st.session_state.evaluated_machines:
        st.info("Todavía no se han añadido máquinas evaluadas.")
        return

    machines_df = pd.DataFrame(st.session_state.evaluated_machines)
    machines_df = machines_df.copy()
    machines_df["failure_probability"] = (machines_df["failure_probability"] * 100).round(1)

    display_df = machines_df.rename(
        columns={
            "machine_id": "ID máquina",
            "machine_type": "Tipo",
            "air_temp_c": "Temp. ambiente (°C)",
            "process_temp_c": "Temp. proceso (°C)",
            "rotational_speed": "Velocidad (rpm)",
            "torque": "Torque (Nm)",
            "tool_wear": "Desgaste (min)",
            "failure_probability": "Probabilidad fallo (%)",
            "risk_level": "Nivel de riesgo",
            "recommendation": "Recomendación",
            "prediction": "Predicción",
            "warning_signals": "Señales"
        }
    )

    st.dataframe(display_df, use_container_width=True)

    csv_data = display_df.to_csv(index=False).encode("utf-8")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="Descargar CSV",
            data=csv_data,
            file_name="maquinas_evaluadas.csv",
            mime="text/csv"
        )

    with col2:
        if st.button("Limpiar tabla"):
            st.session_state.evaluated_machines = []
            st.success("Tabla limpiada. Cambia de sección o recarga para ver la actualización.")

# Resume las métricas principales del modelo final.
def show_model_performance():
    st.title("Rendimiento del modelo")

    st.write(
        """
        El modelo final seleccionado fue Gradient Boosting con un umbral de decisión ajustado a 0.3.

        Este ajuste busca equilibrar la detección de fallos reales con el control de falsas alarmas.
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Precision fallo", "0.80")
    col2.metric("Recall fallo", "0.82")
    col3.metric("F1-score fallo", "0.81")

    st.write(
        """
        - **Precision**: cuando el modelo avisa de fallo, mide cuántas veces acierta.
        - **Recall**: de todos los fallos reales, mide cuántos detecta.
        - **F1-score**: resume el equilibrio entre precision y recall.
        """
    )

# Limpia los nombres técnicos de las variables para mostrarlos mejor en la app.
def clean_feature_name(feature_name):
    return (
        feature_name
        .replace("categorical__", "")
        .replace("numeric__", "")
        .replace("Type_", "Tipo ")
        .replace("Air temperature [K]", "Temperatura ambiente")
        .replace("Process temperature [K]", "Temperatura de proceso")
        .replace("Rotational speed [rpm]", "Velocidad de rotación")
        .replace("Torque [Nm]", "Torque")
        .replace("Tool wear [min]", "Desgaste de herramienta")
    )

# Muestra una explicación global del modelo mediante importancia de variables.
def show_model_explanation(model_artifact):
    st.title("Explicación del modelo")

    st.write(
        """
        Esta sección muestra una explicación global del modelo. No explica una predicción concreta,
        pero ayuda a entender qué variables suelen tener más peso en las decisiones del modelo.
        """
    )

    model = model_artifact["model"]
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    importance_df = pd.DataFrame(
        {
            "variable": [clean_feature_name(name) for name in feature_names],
            "importance": importances
        }
    ).sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=importance_df,
        x="importance",
        y="variable",
        ax=ax
    )
    ax.set_title("Importancia global de variables")
    ax.set_xlabel("Importancia")
    ax.set_ylabel("Variable")
    st.pyplot(fig)

    st.write(
        """
        Esta importancia ayuda a interpretar el comportamiento general del modelo, pero no debe leerse como una causa directa.
        Una variable importante significa que el modelo la usa mucho para separar casos de mayor y menor riesgo.
        """
    )

# Explica las limitaciones actuales de la demo y posibles mejoras futuras.
def show_limitations():
    st.title("Limitaciones y próximos pasos")

    st.subheader("Limitaciones")
    st.write(
        """
        - El dataset utilizado es sintético.
        - La app no está conectada a sensores reales.
        - Las máquinas añadidas se guardan solo durante la sesión.
        - No existe todavía una base de datos real.
        - El modelo debería validarse con datos históricos de una empresa antes de usarse en producción.
        """
    )

    st.subheader("Próximos pasos")
    st.write(
        """
        - Conectar la app a una base de datos.
        - Integrar datos reales de sensores o sistemas de producción.
        - Añadir explicación individual de cada predicción.
        - Monitorizar el rendimiento del modelo con nuevos datos.
        - Reentrenar el modelo periódicamente.
        """
    )

# Controla la navegación principal de la app.
def main():
    initialize_session_state()
    model_artifact = load_model_artifact()

    st.sidebar.title("Navegación")
    section = st.sidebar.radio(
        "Selecciona una sección",
        [
            "Inicio",
            "Evaluar máquina",
            "Máquinas evaluadas",
            "Rendimiento del modelo",
            "Explicación del modelo",
            "Limitaciones"
        ]
    )

    if section == "Inicio":
        show_home()
    elif section == "Evaluar máquina":
        show_machine_evaluation(model_artifact)
    elif section == "Máquinas evaluadas":
        show_evaluated_machines()
    elif section == "Rendimiento del modelo":
        show_model_performance()
    elif section == "Explicación del modelo":
        show_model_explanation(model_artifact)
    elif section == "Limitaciones":
        show_limitations()


if __name__ == "__main__":
    main()
