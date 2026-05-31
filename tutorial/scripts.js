AOS.init({
    duration: 800,
    once: true,
    easing: 'ease-out-quad'
});

lucide.createIcons();

const scrollToTopBtn = document.getElementById("scrollToTopBtn");

window.addEventListener("scroll", () => {
    if (window.scrollY > 300) {
        scrollToTopBtn.classList.remove("opacity-0", "pointer-events-none", "translate-y-4");
        scrollToTopBtn.classList.add("opacity-100", "translate-y-0");
    } else {
        scrollToTopBtn.classList.add("opacity-0", "pointer-events-none", "translate-y-4");
        scrollToTopBtn.classList.remove("opacity-100", "translate-y-0");
    }
});

document.querySelectorAll('pre').forEach(pre => {
    const wrapper = pre.parentElement;
    wrapper.classList.add('code-block-wrapper');
    if (getComputedStyle(wrapper).position === 'static') {
        wrapper.style.position = 'relative';
    }

    const btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.textContent = 'Copy';
    btn.setAttribute('aria-label', 'Copy code to clipboard');

    btn.addEventListener('click', async () => {
        const text = pre.textContent;
        try {
            await navigator.clipboard.writeText(text);
        } catch {
            const ta = document.createElement('textarea');
            ta.value = text;
            ta.style.position = 'fixed';
            ta.style.opacity = '0';
            document.body.appendChild(ta);
            ta.select();
            document.execCommand('copy');
            document.body.removeChild(ta);
        }
        btn.textContent = 'Copied!';
        btn.classList.add('copied');
        setTimeout(() => {
            btn.textContent = 'Copy';
            btn.classList.remove('copied');
        }, 2000);
    });

    wrapper.appendChild(btn);
});
