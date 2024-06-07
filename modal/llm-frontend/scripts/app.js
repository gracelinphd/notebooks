document.getElementById('question-form').addEventListener('submit', async function(event) {
    event.preventDefault();
    
    const question = document.getElementById('question').value;
    const responseContainer = document.getElementById('response-container');
    responseContainer.innerHTML = 'Loading...';

    try {
        const response = await fetch(`/completion/${encodeURIComponent(question)}`);
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let result = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            result += decoder.decode(value, { stream: true });
            responseContainer.innerHTML = result;
        }
    } catch (error) {
        responseContainer.innerHTML = 'Error: ' + error.message;
    }
});