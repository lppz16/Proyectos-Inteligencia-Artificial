# Ficha de análisis

## 1. Nombre del Space

**Nombre:** Ultra-fast AI image generation

**Enlace:** https://huggingface.co/spaces/mrfakename/Z-Image-Turbo
---

## 2. ¿Qué hace el agente?

Este agente crea imagenes a través una descripción dada por nosotros, es decir, el prompt que le enviamos (este debe ser lo más detallado, preciso y coherente posible para lograr una imagen poco ambigüa), y utiliza un modelo llamado Z-Image-Turbo.

---

## 3. Análisis PEAS

| Elemento | Respuesta |
|----------|-----------|
| **Performance** | El agente debe crear imágenes precisas y realistas con base al prompt que le enviamos, esto de manera rápida |
| **Environment** | El agente interactua con la información proporcionada por nosotros en el prompt a través de la página web |
| **Actuators** | El agente crea la imagen que nosotros le solicitamos a través del prompt |
| **Sensors** | El agente recibe como entrada el prompt que le enviamos, de este depende que tan precisa y realista sea la imagen, es decir, si tiene parámetros como estilo, iluminosidad, filtros, etc... |

---

## 4. Clasificación del entorno

| Propiedad | Clasificación | Justificación |
|-----------|---------------|---------------|
| **Observable** | Parcial | Ya que el agente solo conoce la información que nosotros le enviamos a través del prompt, de ahí, que debemos ser lo más detallados y precisos posible para poder obtener la imagen sin ambigüedades |
| **Determinista** | No | Ya que observé que un mismo prompt generó varias imágenes diferentes |
| **Episódico** | Sí | Porque el resultado de nuestros prompts no afectará a los otros |
| **Estático** | Sí | Ya que cuando está generando la imagen, nuestro prompt ya no podrá cambiar |
| **Discreto** | Sí | Ya que nuestros prompts y las imágenes que crea el agente las genera de manera independiente a las demas |
| **Conocido** | Sí | Ya que este agente conoce la entrada, o sea, el prompt, sin embargo, el procedimiento utilizado para generar las imágenes son mediante el modelo de su entrenamiento |

---

## 5. ¿Qué tipo de programa de agente creen que es?

- [ ] Agente de reflejo simple
- [ ] Agente basado en modelo
- [ ] Agente basado en objetivos
- [ ] Agente basado en utilidad
- [X] Agente con aprendizaje

Considero que es u n agente con aprendizaje, ya que para poder llegar a genera imágenes con base a nuestros prompts, este debió haber sido entrenado con una enorme cantidaad de imágenes y prompts, y gracias a estos entrenamientos, hoy en día es capaz de generar las imágenes de manera precisa y coherente.

---

# Discusión en clase

Después de las presentaciones, discutiremos preguntas como:

- ¿Dos Spaces diferentes pueden compartir el mismo tipo de entorno?
- ¿Es posible saber con certeza qué tipo de agente implementa un Space únicamente observándolo?
- ¿Qué diferencia existe entre el comportamiento observable de un agente y su implementación interna?

---

# Reto adicional

Encuentre un Space que pueda clasificarse como:

1. **Totalmente observable, determinista y episódico:**
https://huggingface.co/spaces/Iker/Translate-100-languages ---- **M2M100 Translator:** Considero que este es un buen ejemplo, ya que es totalmente observable, ya que el agente recibe todo el texto que necesita para ser traducido y no necesita más infromación adicional. Además, es determinista porque la traducción que crea es la misma bajo el mismo lenguaje, y episódico porque cada traducción es independiente de las anteriores.

2. **Parcialmente observable, estocástico y secuencial.** 
https://huggingface.co/spaces/MostLime/lcm-chess-playground --- **LCM Chess Playground:** Este, es parcialmente observable porque el agente logra observar el estado actual del trablero y como ha jugado el otro player, sin embargo, no conoce cual será el siguiente movimiento del otro player. Es estocástico, ya que no puede ser 100% predecible, debido a lo anterior dicho. Y, secuencial, porque cada movimiento va a influer en el siguiente, y hasta en el resultado final.

