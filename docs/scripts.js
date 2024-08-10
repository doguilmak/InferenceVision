// Function to toggle the dropdown menu
document.addEventListener('DOMContentLoaded', function () {
    const dropdownToggle = document.querySelector('.dropdown-toggle');
    const dropdownContent = document.querySelector('.dropdown-content');

    dropdownToggle.addEventListener('click', function () {
        const expanded = dropdownToggle.getAttribute('aria-expanded') === 'true';
        dropdownToggle.setAttribute('aria-expanded', !expanded);
        dropdownContent.setAttribute('aria-hidden', expanded);
        dropdownContent.classList.toggle('show');
    });

    document.querySelector('.dropdown-toggle').addEventListener('click', function() {
        var dropdownContent = document.querySelector('.dropdown-content');
        var isExpanded = this.getAttribute('aria-expanded') === 'true';
        
        this.setAttribute('aria-expanded', !isExpanded);
        dropdownContent.setAttribute('aria-hidden', isExpanded);
    });
});

// Function to copy code to clipboard
function copyToClipboard(button) {
    const codeContainer = button.parentElement;
    const code = codeContainer.querySelector('code').textContent;

    navigator.clipboard.writeText(code).then(() => {
        button.textContent = 'Copied!';
        button.setAttribute('aria-live', 'polite');
        setTimeout(() => {
            button.textContent = 'Copy';
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy text: ', err);
        button.textContent = 'Failed';
        setTimeout(() => {
            button.textContent = 'Copy';
        }, 2000);
    });
}

// Accessibility: Close dropdown menu when clicking outside
document.addEventListener('click', function (event) {
    const dropdownContent = document.querySelector('.dropdown-content');
    const dropdownToggle = document.querySelector('.dropdown-toggle');

    if (!dropdownToggle.contains(event.target) && !dropdownContent.contains(event.target)) {
        if (dropdownContent.classList.contains('show')) {
            dropdownContent.classList.remove('show');
            dropdownToggle.setAttribute('aria-expanded', 'false');
            dropdownContent.setAttribute('aria-hidden', 'true');
        }
    }
});
