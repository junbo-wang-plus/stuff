document.getElementById('export').addEventListener('click', async () => {
    console.log('Export button clicked');
    try {
        const response = await chrome.storage.local.get(['clippings']);
        console.log('Current clippings:', response.clippings);
        
        if (!response.clippings) {
            alert('No clippings to export yet!');
            return;
        }

        // Create and trigger download
        const blob = new Blob([response.clippings], {type: 'text/plain'});
        const url = URL.createObjectURL(blob);
        
        chrome.downloads.download({
            url: url,
            filename: 'web_clippings.txt',
            saveAs: true
        }, (downloadId) => {
            if (chrome.runtime.lastError) {
                console.error('Download failed:', chrome.runtime.lastError);
                alert('Export failed: ' + chrome.runtime.lastError.message);
            } else {
                console.log('Download started with ID:', downloadId);
                URL.revokeObjectURL(url);
                window.close();
            }
        });
    } catch (error) {
        console.error('Export error:', error);
        alert('Export failed: ' + error.message);
    }
});

// Log when popup loads
console.log('Popup script loaded');
