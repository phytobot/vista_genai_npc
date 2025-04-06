let chatBoxText;

const config = {
    type: Phaser.AUTO,
    width: 800,
    height: 600,
    backgroundColor: '#242424',
    scene: {
        preload: preload,
        create: create,
        update: update
    }
};

const game = new Phaser.Game(config);

function preload() {
    // Load assets (player, npc, etc.)
    this.load.image('npc', 'assets/npc.png');
}

function create() {
    const scene = this;

    // Add NPC
    const npc = this.add.sprite(400, 300, 'npc').setInteractive();

    // Create chat box UI
    const bg = this.add.rectangle(400, 550, 750, 100, 0x000000, 0.7).setOrigin(0.5);
    chatBoxText = this.add.text(50, 510, 'Click NPC to chat...', {
        fontSize: '18px',
        color: '#FFFFFF',
        wordWrap: { width: 700 }
    });

    npc.on('pointerdown', async () => {
        const userMessage = prompt("You:"); // Replace with input box later if needed.
        if (userMessage) {
            chatBoxText.setText("Talking to NPC...");
            const npcData = await sendMessageToNPC(userMessage);
            console.log("NPC Data:", npcData);
            console.log("Audio URL:", npcData.audio_url);


            chatBoxText.setText("NPC: " + npcData.npc_reply);

            // Play audio if available
            if (npcData.audio_url) {
                const audio = new Audio('http://localhost:5000' + npcData.audio_url);

                audio.play();
            }
        }
    });
}

function update() {}

async function sendMessageToNPC(message) {
    try {
        const response = await fetch('http://localhost:5000/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });

        if (!response.ok) throw new Error('Failed to fetch from backend.');

        const data = await response.json();
        return {
            npc_reply: data.npc_reply,
            audio_url: data.audio_url
        };
    } catch (err) {
        console.error(err);
        return {
            npc_reply: "I don't understand...",
            audio_url: null
        };
    }
}
