export default function bindSasHistory({setTriggerValue}) {
  // Same-page history changes the URL without generating a Streamlit widget event.
  const onHistory = () => setTriggerValue("route", window.location.search);
  window.addEventListener("popstate", onHistory);
  return () => window.removeEventListener("popstate", onHistory);
}
