"use strict";
try {
  const theme = localStorage.getItem("nanoreason-theme");
  if (theme === "light" || theme === "dark")
    document.documentElement.dataset.theme = theme;
} catch (_) {
  /* Use system theme when storage is unavailable. */
}
