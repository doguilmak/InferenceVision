// Header scroll effect
window.addEventListener('scroll', () => {
    const header = document.getElementById('header');
    if (window.scrollY > 50) {
        header.classList.add('scrolled');
    } else {
        header.classList.remove('scrolled');
    }
});

// Copy code functionality
function copyCode(button) {
    const codeElement = button.previousElementSibling;
    const code = codeElement.textContent;

    navigator.clipboard.writeText(code).then(() => {
        const originalText = button.textContent;
        const originalBg = button.style.background;

        button.textContent = 'Copied!';
        button.style.background = '#4caf50';

        setTimeout(() => {
            button.textContent = originalText;
            button.style.background = originalBg;
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy:', err);
        button.textContent = 'Failed!';
        setTimeout(() => {
            button.textContent = 'Copy';
        }, 2000);
    });
}

// Smooth scroll for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            const headerOffset = 100;
            const elementPosition = target.getBoundingClientRect().top;
            const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

            window.scrollTo({
                top: offsetPosition,
                behavior: 'smooth'
            });
        }
    });
});

// Mobile menu toggle (if you want to add mobile menu functionality)
const mobileMenuBtn = document.querySelector('.mobile-menu');
const navUl = document.querySelector('nav ul');

if (mobileMenuBtn && navUl) {
    mobileMenuBtn.addEventListener('click', () => {
        navUl.classList.toggle('open');
    });

    navUl.addEventListener('mouseleave', () => {
        navUl.classList.remove('open');
    });
    document.addEventListener('click', (e) => {
        if (!navUl.contains(e.target) && !mobileMenuBtn.contains(e.target)) {
            navUl.classList.remove('open');
        }
    });
}

// Intersection Observer for fade-in animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('fade-in');
            observer.unobserve(entry.target);
        }
    });
}, observerOptions);

// Observe elements for animation
document.addEventListener('DOMContentLoaded', () => {
    const animateElements = document.querySelectorAll('.grid-item, .content-card, .code-block, .objective-box');
    animateElements.forEach(el => observer.observe(el));
});
