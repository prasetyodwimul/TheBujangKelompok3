const form = document.querySelector('#prediction-form');
if (form) {
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const result = document.querySelector('#result');
    result.textContent = 'Memproses...';
    const payload = Object.fromEntries(new FormData(form).entries());
    try {
      const response = await fetch(`/api/predict/${form.dataset.kind}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      result.textContent = response.ok ? JSON.stringify(data, null, 2) : `Error: ${data.error}`;
    } catch (error) {
      result.textContent = `Tidak dapat menghubungi server: ${error.message}`;
    }
  });
}