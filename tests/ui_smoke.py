import warnings; warnings.filterwarnings("ignore")
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("../app/streamlit_app.py", default_timeout=600)
at.run()
assert not at.exception, at.exception
at.sidebar.selectbox[0].select("Breast cancer (30 features, binary)")
at.run()
assert not at.exception, at.exception
print("target default:", at.sidebar.selectbox[1].value)
at.sidebar.slider[0].set_value(1)
at.sidebar.button[0].click()
at.run()
assert not at.exception, at.exception
print("metrics:", [(m.label, m.value) for m in at.metric])
print("tabs:", [t.label for t in at.tabs])
print("OK")
