AOS.init({
    duration: 800,
    once: true,
    easing: 'ease-out-quad'
});

lucide.createIcons();

// Video player
const videoContainer = document.getElementById('demo-video-container');
const video = document.getElementById('demo-video');
const overlay = document.getElementById('demo-video-overlay');
const playBtn = document.getElementById('play-btn');
const playIcon = document.getElementById('play-icon');
const pauseIcon = document.getElementById('pause-icon');
const rewindBtn = document.getElementById('rewind-btn');
const forwardBtn = document.getElementById('forward-btn');
const progressBar = document.getElementById('progress-bar');
const progressFill = document.getElementById('progress-fill');
const timeDisplay = document.getElementById('time-display');

function formatTime(s) {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, '0')}`;
}

function togglePlay() {
    if (video.paused) {
        overlay.classList.add('opacity-0', 'pointer-events-none');
        video.play();
    } else {
        video.pause();
        overlay.classList.remove('opacity-0', 'pointer-events-none');
    }
}

function updatePlayBtn() {
    if (video.paused) {
        playIcon.classList.remove('hidden');
        pauseIcon.classList.add('hidden');
    } else {
        playIcon.classList.add('hidden');
        pauseIcon.classList.remove('hidden');
    }
}

function updateProgress() {
    if (!video.duration) return;
    const pct = (video.currentTime / video.duration) * 100;
    progressFill.style.width = pct + '%';
    timeDisplay.textContent = `${formatTime(video.currentTime)} / ${formatTime(video.duration)}`;
}

videoContainer.addEventListener('click', (e) => {
    if (e.target.closest('#demo-video-controls') || e.target.closest('#progress-bar')) return;
    togglePlay();
});

playBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    togglePlay();
});

rewindBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    video.currentTime = Math.max(0, video.currentTime - 10);
});

forwardBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    video.currentTime = Math.min(video.duration, video.currentTime + 10);
});

progressBar.addEventListener('click', (e) => {
    e.stopPropagation();
    const rect = progressBar.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    video.currentTime = pct * video.duration;
});

video.addEventListener('timeupdate', updateProgress);
video.addEventListener('play', updatePlayBtn);
video.addEventListener('pause', updatePlayBtn);
video.addEventListener('ended', () => {
    overlay.classList.remove('opacity-0', 'pointer-events-none');
    updatePlayBtn();
    updateProgress();
});

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
