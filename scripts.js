function showLoginForm() {
    document.getElementById("loginForm").style.display = "block";
    document.getElementById("registrationForm").style.display = "none";
}

function showRegistrationForm() {
    document.getElementById("loginForm").style.display = "none";
    document.getElementById("registrationForm").style.display = "block";
}

document.addEventListener("DOMContentLoaded", function () {
    // Handle login form submission
    const loginForm = document.getElementById("login-form");
    if (loginForm) {
        loginForm.addEventListener("submit", function (e) {
            e.preventDefault(); // Prevent default form submission
            
            const formData = new FormData(this);
            
            fetch('login_register.php', {
                method: 'POST',
                body: formData
            })
            .then(response => response.text())
            .then(data => {
                if (data.includes("Login successful")) {
                    window.location.href = "dashboard.html";
                } else {
                    document.getElementById("loginMessage").innerHTML = 
                        '<div class="error-message">' + data + '</div>';
                }
            })
            .catch(error => {
                console.error('Error:', error);
            });
        });
    }

    // Handle registration form submission
    const registrationForm = document.getElementById("register-form");
    if (registrationForm) {
        registrationForm.addEventListener("submit", function (e) {
            e.preventDefault(); // Prevent default form submission
            
            const formData = new FormData(this);
            
            fetch('login_register.php', {
                method: 'POST',
                body: formData
            })
            .then(response => response.text())
            .then(data => {
                document.getElementById("registrationMessage").innerHTML = 
                    '<div class="' + (data.includes("successful") ? 'success-message' : 'error-message') + '">' + 
                    data + '</div>';
                    
                if (data.includes("Registration successful")) {
                    // Clear the form
                    registrationForm.reset();
                    // Switch to login form after successful registration
                    setTimeout(() => {
                        document.getElementById("login").checked = true;
                        showLoginForm();
                    }, 2000);
                }
            })
            .catch(error => {
                console.error('Error:', error);
            });
        });
    }
});