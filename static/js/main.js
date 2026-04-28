// Image preview + drag-drop
const fileInput = document.getElementById('file');
const uploadBox = document.getElementById('uploadBox');
const placeholder = document.getElementById('placeholder');
const preview = document.getElementById('preview');

if (fileInput) {
  fileInput.addEventListener('change', e => showPreview(e.target.files[0]));

  ['dragover', 'dragleave', 'drop'].forEach(ev => {
    uploadBox.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); });
  });
  uploadBox.addEventListener('drop', e => {
    const f = e.dataTransfer.files[0];
    if (f) { fileInput.files = e.dataTransfer.files; showPreview(f); }
  });
}

function showPreview(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    preview.src = e.target.result;
    preview.hidden = false;
    placeholder.style.display = 'none';
  };
  reader.readAsDataURL(file);
}
