document.addEventListener('DOMContentLoaded', () => {
    const chatOutput = document.getElementById('chat-output');
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const sessionIdInput = document.getElementById('session-id');

    // Function to add a message to the chat output
    function addMessageToChat(sender, message, isUser) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        messageDiv.classList.add(isUser ? 'user-message' : 'bot-message');

        if (isUser) {
            messageDiv.textContent = message;
        } else {
            // For bot messages, parse Markdown and sanitize the HTML
            if (window.marked && window.DOMPurify) {
                // Ensure marked is called as a function if using the global `marked()`
                // For newer versions (e.g., v4+), it might be `marked.parse()`
                // Checking for both for broader compatibility with CDN versions.
                let rawHtml;
                if (typeof window.marked === 'function') {
                    rawHtml = window.marked(message); // Older versions or specific CDN builds
                } else if (window.marked && typeof window.marked.parse === 'function') {
                    rawHtml = window.marked.parse(message); // Newer versions (marked.parse())
                } else {
                    console.warn('marked.js is loaded, but the parsing function (marked() or marked.parse()) was not found. Falling back.');
                    messageDiv.innerHTML = message.replace(/\\n/g, '<br>'); // Fallback
                    chatOutput.appendChild(messageDiv);
                    chatOutput.scrollTop = chatOutput.scrollHeight;
                    return;
                }
                messageDiv.innerHTML = window.DOMPurify.sanitize(rawHtml);
            } else {
                // Fallback for basic newline handling if libraries are not loaded
                console.warn('marked.js or DOMPurify not loaded. Falling back to basic newline replacement.');
                messageDiv.innerHTML = message.replace(/\\n/g, '<br>');
            }
        }

        chatOutput.appendChild(messageDiv);
        chatOutput.scrollTop = chatOutput.scrollHeight; // Scroll to bottom
    }

    // Handle sending a message
    async function sendMessage() {
        const message = messageInput.value.trim();
        const sessionId = sessionIdInput.value.trim();

        if (!message) return;
        if (!sessionId) {
            alert('Please enter a Session ID.');
            return;
        }

        addMessageToChat('User', message, true);
        messageInput.value = '';
        sendButton.disabled = true;

        // Add a placeholder for the bot's response
        const botMessageDiv = document.createElement('div');
        botMessageDiv.classList.add('message', 'bot-message');
        chatOutput.appendChild(botMessageDiv);
        chatOutput.scrollTop = chatOutput.scrollHeight;

        let botMessageContent = '';

        try {
            const response = await fetch('http://127.0.0.1:8000/chat', { // Corrected URL for local streaming
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message, session_id: sessionId }),
            });

            if (!response.ok || !response.body) {
                const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
                botMessageDiv.innerHTML = `Error: ${response.status} - ${errorData.detail || 'Could not fetch response.'}`;
                throw new Error(`API request failed with status ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) {
                    break;
                }

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop(); // Keep any partial line for the next chunk

                for (const line of lines) {
                    if (line.trim() === '') continue;
                    try {
                        const chunk = JSON.parse(line);
                        if (chunk.response) {
                            botMessageContent += chunk.response;
                            // Progressively render markdown
                            if (window.marked && window.DOMPurify) {
                                const rawHtml = window.marked.parse(botMessageContent);
                                botMessageDiv.innerHTML = window.DOMPurify.sanitize(rawHtml);
                            } else {
                                // Fallback to simple text, escaping HTML
                                botMessageDiv.textContent = botMessageContent;
                            }
                            chatOutput.scrollTop = chatOutput.scrollHeight;
                        }
                    } catch (e) {
                        console.error('Failed to parse JSON chunk:', line, e);
                    }
                }
            }
        } catch (error) {
            console.error('Streaming failed:', error);
            if (!botMessageContent) { // If we never got any content, show an error message.
                 botMessageDiv.innerHTML = 'Failed to connect to the server. Please check the console for details.';
            }
        } finally {
            sendButton.disabled = false;
            messageInput.focus();
        }
    }

    sendButton.addEventListener('click', sendMessage);
    messageInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

    sessionIdInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            messageInput.focus(); // Move to message input on enter
        }
    });
});
