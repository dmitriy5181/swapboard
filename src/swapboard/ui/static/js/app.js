(function () {
    const STORAGE_KEY = "swapboard-theme";
    const toggle = document.getElementById("theme-toggle");
    const icon = document.getElementById("theme-toggle-icon");

    function apply(mode) {
        document.documentElement.setAttribute("data-bs-theme", mode);
        if (icon) {
            icon.className =
                mode === "dark" ? "bi bi-sun-fill" : "bi bi-moon-stars-fill";
        }
    }

    apply(document.documentElement.getAttribute("data-bs-theme") || "light");

    if (!toggle) {
        return;
    }

    toggle.addEventListener("click", function () {
        const next =
            document.documentElement.getAttribute("data-bs-theme") === "dark"
                ? "light"
                : "dark";
        apply(next);
        try {
            localStorage.setItem(STORAGE_KEY, next);
        } catch (e) {
            // Ignore browsers/environments where localStorage is unavailable.
        }
    });
})();
