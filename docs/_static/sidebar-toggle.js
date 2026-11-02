// Workaround for sphinx-book-theme >= 1.2 / pydata-sphinx-theme >= 0.16: both themes attach their sidebar toggle
// handlers only to the first `.primary-toggle` / `.secondary-toggle` element on the page, which is the (hidden)
// button in the top navbar. The visible "Toggle primary sidebar" button in the article header therefore does nothing.
// Clicks on any other toggle button are forwarded to the first one, which has the handlers attached. The listener
// runs in the capture phase on the document and stops the original event, so the toggle never fires twice.
document.addEventListener(
  "click",
  (event) => {
    for (const selector of [".primary-toggle", ".secondary-toggle"]) {
      const button = event.target.closest(selector);
      const first = document.querySelector(selector);
      if (button && first && button !== first) {
        event.preventDefault();
        event.stopImmediatePropagation();
        first.click();
        return;
      }
    }
  },
  true,
);
