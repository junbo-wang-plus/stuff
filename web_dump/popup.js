// Load saved clippings when popup opens
document.addEventListener('DOMContentLoaded', async () => {
    const editor = document.getElementById('editor');
    try {
        const result = await chrome.storage.local.get(['clippings']);
        editor.value = result.clippings || '';
    } catch (error) {
        console.error('Failed to load clippings:', error);
        editor.value = 'Error loading clippings.';
    }
});

// Save changes to storage
document.getElementById('save').addEventListener('click', async () => {
    const editor = document.getElementById('editor');
    try {
        await chrome.storage.local.set({ clippings: editor.value });
        // Visual feedback
        const button = document.getElementById('save');
        button.textContent = 'Saved!';
        setTimeout(() => {
            button.textContent = 'Save Changes';
        }, 1000);
    } catch (error) {
        console.error('Failed to save:', error);
        alert('Failed to save changes.');
    }
});

// Export to file
document.getElementById('export').addEventListener('click', async () => {
    const editor = document.getElementById('editor');
    try {
        const blob = new Blob([editor.value], {type: 'text/plain'});
        const url = URL.createObjectURL(blob);
        
        await chrome.downloads.download({
            url: url,
            filename: 'web_clippings.txt',
            saveAs: true
        });
        
        URL.revokeObjectURL(url);
    } catch (error) {
        console.error('Export failed:', error);
        alert('Failed to export file.');
    }
});

// Clear all clippings
document.getElementById('clear').addEventListener('click', async () => {
    if (confirm('Are you sure you want to clear all clippings? This cannot be undone.')) {
        try {
            await chrome.storage.local.remove(['clippings']);
            document.getElementById('editor').value = '';
        } catch (error) {
            console.error('Failed to clear:', error);
            alert('Failed to clear clippings.');
        }
    }
});

// Auto-save when popup closes
window.addEventListener('unload', () => {
    const editor = document.getElementById('editor');
    chrome.storage.local.set({ clippings: editor.value });
});
