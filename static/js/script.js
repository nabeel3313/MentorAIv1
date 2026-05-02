// General JavaScript functionality for DirectProf

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Auto-dismiss alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            if (alert.classList.contains('show')) {
                const bsAlert = new bootstrap.Alert(alert);
                bsAlert.close();
            }
        }, 5000);
    });

    // Form validation enhancements
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            const submitBtn = this.querySelector('button[type="submit"], input[type="submit"]');
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Loading...';
            }
        });
    });

    // Password strength indicator
    const passwordInputs = document.querySelectorAll('input[type="password"]');
    passwordInputs.forEach(input => {
        input.addEventListener('input', function() {
            const strengthIndicator = this.parentNode.querySelector('.password-strength');
            if (strengthIndicator) {
                const strength = calculatePasswordStrength(this.value);
                updatePasswordStrengthIndicator(strengthIndicator, strength);
            }
        });
    });
});

function calculatePasswordStrength(password) {
    let strength = 0;
    if (password.length >= 8) strength++;
    if (password.match(/[a-z]/) && password.match(/[A-Z]/)) strength++;
    if (password.match(/\d/)) strength++;
    if (password.match(/[^a-zA-Z\d]/)) strength++;
    return strength;
}

function updatePasswordStrengthIndicator(indicator, strength) {
    const classes = ['bg-danger', 'bg-warning', 'bg-info', 'bg-success'];
    const texts = ['Very Weak', 'Weak', 'Good', 'Strong'];
    
    indicator.className = 'progress-bar';
    indicator.classList.add(classes[strength] || 'bg-danger');
    indicator.style.width = `${(strength / 3) * 100}%`;
    indicator.textContent = texts[strength] || 'Very Weak';
}

// Session management functions
function joinSession(sessionId) {
    // Implementation for joining a session
    console.log('Joining session:', sessionId);
    window.location.href = `/session/${sessionId}`;
}

function cancelSession(sessionId) {
    if (confirm('Are you sure you want to cancel this session?')) {
        // Implementation for canceling a session
        console.log('Canceling session:', sessionId);
    }
}

// Chat functionality
function sendChatMessage(message, sessionId) {
    // Implementation for sending chat messages
    console.log('Sending message:', message, 'to session:', sessionId);
}

// Video session controls
function toggleVideo() {
    // Implementation for toggling video
    console.log('Toggling video');
}

function toggleAudio() {
    // Implementation for toggling audio
    console.log('Toggling audio');
}

function toggleScreenShare() {
    // Implementation for screen sharing
    console.log('Toggling screen share');
}

// Utility functions
function formatDate(dateString) {
    const options = { year: 'numeric', month: 'short', day: 'numeric' };
    return new Date(dateString).toLocaleDateString(undefined, options);
}

function formatTime(dateString) {
    const options = { hour: '2-digit', minute: '2-digit' };
    return new Date(dateString).toLocaleTimeString(undefined, options);
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}