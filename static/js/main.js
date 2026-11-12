/**
 * ResearchMate - Core Frontend Utilities
 */

// Global Toast Notification Helper
function showToast(message, type = "info") {
  const toastContainer = document.getElementById("toast-container");
  if (!toastContainer) return;

  const bgClasses = {
    success: "bg-success text-white",
    danger: "bg-danger text-white",
    warning: "bg-warning text-dark",
    info: "bg-primary text-white",
  };

  const id = "toast-" + Date.now();
  const html = `
    <div id="${id}" class="toast align-items-center ${bgClasses[type] || bgClasses.info} border-0 shadow" role="alert" aria-live="assertive" aria-atomic="true">
      <div class="d-flex">
        <div class="toast-body py-2 px-3 fw-medium">
          ${message}
        </div>
        <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
      </div>
    </div>
  `;

  toastContainer.insertAdjacentHTML("beforeend", html);
  const toastEl = document.getElementById(id);
  const toast = new bootstrap.Toast(toastEl, { delay: 4000 });
  toast.show();
  toastEl.addEventListener("hidden.bs.toast", () => toastEl.remove());
}

// Copy to Clipboard Helper
function copyToClipboard(text, successMsg = "Copied to clipboard!") {
  if (!navigator.clipboard) {
    const el = document.createElement("textarea");
    el.value = text;
    document.body.appendChild(el);
    el.select();
    document.execCommand("copy");
    document.body.removeChild(el);
    showToast(successMsg, "success");
    return;
  }
  navigator.clipboard.writeText(text).then(() => {
    showToast(successMsg, "success");
  }).catch(err => {
    showToast("Failed to copy text: " + err, "danger");
  });
}

// Lightweight Markdown Renderer for Academic Text
function renderMarkdown(md) {
  if (!md) return "";
  let html = md;

  // Escape HTML entities
  html = html.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  // Headers
  html = html.replace(/^### (.*$)/gim, '<h4 class="mt-3 mb-2 fw-bold text-dark">$1</h4>');
  html = html.replace(/^## (.*$)/gim, '<h3 class="mt-4 mb-2 fw-bold border-bottom pb-1 text-primary">$1</h3>');
  html = html.replace(/^# (.*$)/gim, '<h2 class="mt-4 mb-3 fw-bold text-dark text-center">$1</h2>');

  // Bold & Italics
  html = html.replace(/\*\*\*(.*?)\*\*\*/gim, '<strong><em>$1</em></strong>');
  html = html.replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>');
  html = html.replace(/\*(.*?)\*/gim, '<em>$1</em>');

  // Callouts & Warning tags
  html = html.replace(/\[REQUIRES HUMAN VERIFICATION\]/g, '<span class="badge bg-warning text-dark me-1"><i class="bi bi-exclamation-triangle"></i> Requires Human Verification</span>');
  html = html.replace(/\[ORIGINAL RESEARCH REQUIRED:(.*?)\]/g, '<div class="research-alert-marker"><i class="bi bi-pencil-square me-2"></i><strong>ORIGINAL RESEARCH REQUIRED:</strong>$1</div>');

  // Markdown Tables
  html = html.replace(/\|(.+)\|/g, function(match) {
    return match; // Simple preservation, formatted below
  });

  // Blockquotes
  html = html.replace(/^\> (.*$)/gim, '<blockquote class="blockquote ps-3 border-start border-3 border-primary text-muted my-2">$1</blockquote>');

  // Unordered Lists
  html = html.replace(/^\- (.*$)/gim, '<li class="ms-3 mb-1">$1</li>');

  // Newlines to line breaks or paragraphs
  html = html.replace(/\n\n/g, '<br><br>');

  return html;
}

// Sidebar toggle for mobile devices
document.addEventListener("DOMContentLoaded", () => {
  const toggleBtn = document.getElementById("sidebar-toggle");
  const sidebar = document.getElementById("sidebar");
  if (toggleBtn && sidebar) {
    toggleBtn.addEventListener("click", () => {
      sidebar.classList.toggle("show");
    });
  }
});
