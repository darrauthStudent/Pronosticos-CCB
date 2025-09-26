import pandas as pd
import streamlit as st
import hashlib
from src.etl import load_dict_from_feather

# Configuracion de página
st.set_page_config(page_title="Análisis de Series Temporales", layout="wide")

# --- Seguridad muy básica (ejemplo) ---
# ⚠️ Reemplaza por tu propio mecanismo (BD, secrets, etc.)
# Guarda contraseñas como SHA256 de texto plano solo para demo.
# En producción usa bcrypt/argon2 y almacén seguro.


USERS = {
    "admin": hashlib.sha256("1234".encode()).hexdigest(),
    "david": hashlib.sha256("abcd".encode()).hexdigest(),
}

def _hash(pwd: str) -> str:
    return hashlib.sha256(pwd.encode()).hexdigest()

def ensure_session_keys():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "diccionario_datos" not in st.session_state:
        st.session_state.diccionario_datos = None

@st.cache_data
def load_data(_manifest_mtime=None):
    """Carga los datos desde archivos Feather usando cache para optimizar rendimiento"""
    import os
    try:
        manifest_path = os.path.normpath("data/feather_manifest.json")
        diccionario_datos = load_dict_from_feather(manifest_path)
        return diccionario_datos
    except FileNotFoundError as e:
        st.error(f"Error cargando datos: {e}")
        st.error("Asegurate de ejecutar el notebook de limpieza de datos primero.")
        return None
    except Exception as e:
        st.error(f"Error inesperado cargando datos: {e}")
        return None

def get_manifest_mtime():
    """Obtiene la fecha de modificación del archivo manifest para invalidar cache"""
    import os
    try:
        manifest_path = os.path.normpath("data/feather_manifest.json")
        return os.path.getmtime(manifest_path)
    except:
        return 0

def logout():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.success("Sesión cerrada.")
    st.rerun()

def login_ui():
    st.title("🔐 Iniciar sesión")
    with st.form("login_form", clear_on_submit=False):
        user = st.text_input("Usuario")
        pwd = st.text_input("Contraseña", type="password")
        submitted = st.form_submit_button("Ingresar")
    if submitted:
        if user in USERS and USERS[user] == _hash(pwd):
            st.session_state.logged_in = True
            st.session_state.username = user
            st.success("Login exitoso ✅")
            st.rerun()
        else:
            st.error("Credenciales inválidas")

def main():
    ensure_session_keys()

    # Si NO está logueado, muestra login y detén la app
    if not st.session_state.logged_in:
        login_ui()
        st.stop()

    # Sidebar con info de sesión y logout
    with st.sidebar:
        st.caption("Sesión")
        st.success(f"Conectado como **{st.session_state.username}**")
        if st.button("Cerrar sesión"):
            logout()
        
        # Información del estado de los datos
        st.markdown("---")
        st.caption("Estado de los datos")
        if st.session_state.data_loaded:
            st.success("Datos cargados correctamente")
            if st.session_state.diccionario_datos:
                st.info(f"Series disponibles: {len(st.session_state.diccionario_datos)}")
                # Mostrar las series disponibles
                for nombre_serie in st.session_state.diccionario_datos.keys():
                    filas = len(st.session_state.diccionario_datos[nombre_serie])
                    st.caption(f"• {nombre_serie}: {filas} registros")
            
            # Botón para recargar datos
            if st.button("🔄 Recargar datos"):
                # Limpiar cache y session state
                load_data.clear()
                st.session_state.data_loaded = False
                st.session_state.diccionario_datos = None
                st.success("Cache limpiado. Recargando datos...")
                st.rerun()
        else:
            st.warning("Datos no cargados")

    # ================== Contenido de tu app (protegido) ==================
    st.title("Bienvenido a Cronos")
    st.subheader("Explora, descompón y modela tus series temporales")

    # Cargar datos si no están cargados
    if not st.session_state.data_loaded:
        with st.spinner("Cargando datos del sistema..."):
            manifest_mtime = get_manifest_mtime()
            diccionario_datos = load_data(_manifest_mtime=manifest_mtime)
            if diccionario_datos is not None:
                st.session_state.diccionario_datos = diccionario_datos
                st.session_state.data_loaded = True
                st.success("Datos cargados exitosamente!")
                st.rerun()
            else:
                st.error("No se pudieron cargar los datos. Revisa la configuración.")
                st.stop()

    st.write("""
    Aquí podrás:
    - **Explorar la serie de tiempo** y analizar su estacionalidad.
    - **Descomponer la serie** en sus componentes fundamentales.
    - **Aplicar modelos** y visualizar predicciones.
    - **Ver un resumen estadístico** de las pruebas realizadas.
    """)


    st.markdown("---")
    st.info("Desarrollado con Streamlit | Visítanos en: [www.miempresa.com](https://www.miempresa.com)")

if __name__ == "__main__":
    main()
