import pandas as pd
import streamlit as st
import hashlib
import os
from pathlib import Path


def load_dict_from_csv(base_path="data/csv"):
    """
    Carga todos los archivos CSV de una carpeta en un diccionario de DataFrames.
    """
    # Normalizar la ruta base
    base_path = os.path.normpath(base_path)
    
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"No se encontró la carpeta: {base_path}")
    
    data_dict = {}
    csv_files = Path(base_path).glob("*.csv")
    
    for csv_file in csv_files:
        # El nombre del dataset será el nombre del archivo sin extensión
        name = csv_file.stem
        
        # Cargar DataFrame desde CSV
        df = pd.read_csv(csv_file, encoding='utf-8')
        
        # Convertir la columna Fecha a datetime si existe
        if 'Fecha' in df.columns:
            df['Fecha'] = pd.to_datetime(df['Fecha'])
        
        data_dict[name] = df
    
    if not data_dict:
        raise FileNotFoundError(f"No se encontraron archivos CSV en: {base_path}")
    
    return data_dict


# Configuracion de página
st.set_page_config(page_title="🏠 Análisis de Series Temporales", page_icon="🏠", layout="wide")

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
def load_data(_data_folder_mtime=None):
    """Carga los datos desde archivos CSV usando cache para optimizar rendimiento"""
    try:
        diccionario_datos = load_dict_from_csv("data/csv")
        return diccionario_datos
    except FileNotFoundError as e:
        st.error(f"Error cargando datos: {e}")
        st.error("Asegurate de ejecutar el notebook de limpieza de datos primero.")
        return None
    except Exception as e:
        st.error(f"Error inesperado cargando datos: {e}")
        return None

def get_data_folder_mtime():
    """Obtiene la fecha de modificación de la carpeta de datos para invalidar cache"""
    try:
        csv_folder = os.path.normpath("data/csv")
        if os.path.exists(csv_folder):
            # Obtener el tiempo de modificación más reciente de todos los archivos CSV
            max_mtime = 0
            for file in os.listdir(csv_folder):
                if file.endswith('.csv'):
                    file_path = os.path.join(csv_folder, file)
                    max_mtime = max(max_mtime, os.path.getmtime(file_path))
            return max_mtime
        return 0
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
            data_folder_mtime = get_data_folder_mtime()
            diccionario_datos = load_data(_data_folder_mtime=data_folder_mtime)
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
    st.info("Desarrollado con Streamlit | Visítamos en: [www.miempresa.com](https://www.miempresa.com)")

if __name__ == "__main__":
    main()