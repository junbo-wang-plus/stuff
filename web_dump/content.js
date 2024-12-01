let selectionButton = null;
let lastUrl = '';

function createSelectionButton() {
    if (!selectionButton) {
        selectionButton = document.createElement('button');
        selectionButton.id = 'text-clipper-button';
        selectionButton.textContent = '📋';
        selectionButton.className = 'text-clipper-button';
        document.body.appendChild(selectionButton);
    }
}

function updateButtonPosition() {
    const selection = window.getSelection();
    if (!selection.toString()) {
        if (selectionButton) {
            selectionButton.style.display = 'none';
        }
        return;
    }

    const range = selection.getRangeAt(0);
    const rect = range.getBoundingClientRect();
    
    createSelectionButton();
    
    selectionButton.style.display = 'block';
    selectionButton.style.top = `${window.scrollY + rect.bottom + 10}px`;
    selectionButton.style.left = `${window.scrollX + rect.left}px`;
}

async function handleButtonClick() {
    const selection = window.getSelection().toString();
    if (selection) {
        selectionButton.textContent = '⏳';
        
        try {
            const response = await chrome.runtime.sendMessage({
                type: 'saveText',
                text: selection,
                url: window.location.href,
                lastUrl: lastUrl
            });
            
            if (response.success) {
                selectionButton.textContent = '✓';
                lastUrl = window.location.href;
            } else {
                throw new Error('Save failed');
            }
        } catch (error) {
            console.error('Error:', error);
            selectionButton.textContent = '❌';
        }
        
        setTimeout(() => {
            selectionButton.textContent = '📋';
        }, 1000);
    }
}

window.addEventListener('mouseup', updateButtonPosition);
window.addEventListener('selectionchange', updateButtonPosition);

document.addEventListener('click', (event) => {
    if (event.target.id === 'text-clipper-button') {
        handleButtonClick();
    }
});
