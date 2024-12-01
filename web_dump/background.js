let lastUrl = '';

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === 'saveText') {
        console.log('Saving new text clip');
        
        const currentDate = new Date().toISOString().slice(0, 10);
        const currentUrl = message.url;
        let newContent = '';
        
        // Format new content
        if (currentUrl !== lastUrl) {
            newContent = `\n## ${currentDate}\n\nSource: [${currentUrl}](${currentUrl})\n\n`;
            lastUrl = currentUrl;
        }
        newContent += `> ${message.text}\n\n`;

        // Append to stored content
        chrome.storage.local.get(['clippings'], function(result) {
            console.log('Current storage:', result);
            const existingContent = result.clippings || '';
            const updatedContent = existingContent + newContent;
            
            chrome.storage.local.set({ clippings: updatedContent }, function() {
                console.log('Saved to storage:', updatedContent);
                sendResponse({ success: true });
            });
        });

        return true;
    }
});
