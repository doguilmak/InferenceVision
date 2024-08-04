// Function to toggle the dropdown menu
document.addEventListener('DOMContentLoaded', function () {
    const dropdownToggle = document.querySelector('.dropdown-toggle');
    const dropdownContent = document.querySelector('.dropdown-content');

    dropdownToggle.addEventListener('click', function () {
        dropdownContent.classList.toggle('show');
    });
});

// Function to copy code to clipboard
function copyToClipboard(button) {
    const codeContainer = button.parentElement;
    const code = codeContainer.querySelector('code').textContent;
    navigator.clipboard.writeText(code).then(() => {
        button.textContent = 'Copied!';
        setTimeout(() => {
            button.textContent = 'Copy';
        }, 2000);
    });
}
