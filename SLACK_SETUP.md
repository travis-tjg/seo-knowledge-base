# Slack Bot Setup Guide

This guide walks you through setting up the SEO Knowledge Base Slack bot.

## Overview

The bot allows your team to query the SEO knowledge base directly from Slack using:
- **@mentions**: `@SEOBot what is local SEO?`
- **Slash commands**: `/ask what is local SEO?` and `/seo-stats`

## Step 1: Create a Slack App

1. Go to [https://api.slack.com/apps](https://api.slack.com/apps)
2. Click **"Create New App"**
3. Choose **"From scratch"**
4. Enter:
   - **App Name**: `SEO Knowledge Base` (or your preferred name)
   - **Workspace**: Select your workspace
5. Click **"Create App"**

## Step 2: Configure Bot Permissions (OAuth & Permissions)

1. In the left sidebar, click **"OAuth & Permissions"**
2. Scroll to **"Scopes"** section
3. Under **"Bot Token Scopes"**, add these scopes:

| Scope | Purpose |
|-------|---------|
| `app_mentions:read` | Respond when @mentioned |
| `channels:history` | Read messages in channels (for `ingest:` prefix) |
| `channels:read` | Access channel metadata |
| `chat:write` | Send messages |
| `commands` | Handle slash commands |
| `groups:history` | Read messages in private channels (optional) |
| `im:history` | Read DM history |
| `im:read` | Access DM metadata |
| `im:write` | Send DMs |
| `reactions:write` | Add emoji reactions to show processing status |

## Step 3: Enable Socket Mode

Socket Mode lets your bot connect without a public URL.

1. In the left sidebar, click **"Socket Mode"**
2. Toggle **"Enable Socket Mode"** to ON
3. You'll be prompted to create an App-Level Token:
   - **Token Name**: `socket-mode-token`
   - **Scopes**: Add `connections:write`
   - Click **"Generate"**
4. **SAVE THIS TOKEN** - it starts with `xapp-` (this is your `SLACK_APP_TOKEN`) xapp-1-A0A4Y1LGSUB-10171024834629-aa7ff2bcf887d4211e6c6de5c299adcbdcf9d37c6c2ae2a9b611964f7d77249e

## Step 4: Enable Event Subscriptions

1. In the left sidebar, click **"Event Subscriptions"**
2. Toggle **"Enable Events"** to ON
3. Under **"Subscribe to bot events"**, add:
   - `app_mention` - When someone @mentions your bot
   - `message.im` - Direct messages to your bot
   - `message.channels` - Messages in public channels (needed for `ingest:` prefix)
   - `message.groups` - Messages in private channels (optional)

## Step 5: Create Slash Commands

1. In the left sidebar, click **"Slash Commands"**
2. Click **"Create New Command"**

**First command - /ask:**
- Command: `/ask`
- Short Description: `Ask the SEO knowledge base a question`
- Usage Hint: `[your question]`
- Click **"Save"**

**Second command - /seo-stats:**
- Click **"Create New Command"** again
- Command: `/seo-stats`
- Short Description: `Show SEO knowledge base statistics`
- Leave Usage Hint empty
- Click **"Save"**

**Third command - /ingest:**
- Click **"Create New Command"** again
- Command: `/ingest`
- Short Description: `Add a URL to the SEO knowledge base`
- Usage Hint: `[url]`
- Click **"Save"**

## Step 6: Install App to Workspace

1. In the left sidebar, click **"Install App"**
2. Click **"Install to Workspace"**
3. Review permissions and click **"Allow"**
4. **SAVE THE BOT TOKEN** - it starts with `xoxb-` (this is your `SLACK_BOT_TOKEN`) xoxb-28337015505-10159008461575-LE6J2OjzIrweIanlZqrfyyFi

## Step 7: Get Signing Secret

1. In the left sidebar, click **"Basic Information"**
2. Scroll to **"App Credentials"**
3. Find **"Signing Secret"** and click **"Show"**
4. **SAVE THIS SECRET** (this is your `SLACK_SIGNING_SECRET`) b48bade7d780c002723ca4c89ff7a4dd

## Step 8: Configure Environment Variables

Add these to your `.env` file in the seo-knowledge-base directory:

```bash
# Slack Bot Configuration
SLACK_BOT_TOKEN=xoxb-28337015505-10159008461575-LE6J2OjzIrweIanlZqrfyyFi
SLACK_APP_TOKEN=xapp-1-A0A4Y1LGSUB-10171024834629-aa7ff2bcf887d4211e6c6de5c299adcbdcf9d37c6c2ae2a9b611964f7d77249e
SLACK_SIGNING_SECRET=b48bade7d780c002723ca4c89ff7a4dd
```

Or export them in your terminal:

```bash
export SLACK_BOT_TOKEN="xoxb-..."
export SLACK_APP_TOKEN="xapp-..."
export SLACK_SIGNING_SECRET="..."
```

## Step 9: Install Dependencies

```bash
cd /path/to/seo-knowledge-base
source venv/bin/activate
pip install slack-bolt aiohttp
```

## Step 10: Run the Bot

```bash
cd /path/to/seo-knowledge-base
source venv/bin/activate
python slack_bot.py
```

You should see:
```
Starting SEO Knowledge Base Slack Bot...
Connected to knowledge base: 41000 chunks
Bot is ready! Try @mentioning it or using /ask
```

## Using the Bot

### @Mention
In any channel where the bot is present:
```
@SEO Knowledge Base What are the best practices for Google Business Profile?
```

### Slash Command
From any channel:
```
/ask What is topical authority and why does it matter?
```

### Statistics
```
/seo-stats
```

### Direct Message
You can also DM the bot directly - no @ mention needed.

### Ingest URLs (Add content from your phone!)
There are two ways to add URLs to the knowledge base:

**Using the slash command:**
```
/ingest https://moz.com/learn/seo/what-is-seo
```

**Using message prefix (works in any channel the bot is in):**
```
ingest: https://example.com/seo-article
add: https://example.com/another-article
```

The bot will:
1. React with a hourglass while processing
2. Fetch the webpage content
3. Extract the main text
4. Add it to the knowledge base
5. React with a checkmark when done

This is perfect for quickly adding articles you find while browsing on your phone!

## Troubleshooting

### "Missing required environment variables"
Make sure all three environment variables are set:
- `SLACK_BOT_TOKEN` (starts with `xoxb-`)
- `SLACK_APP_TOKEN` (starts with `xapp-`)
- `SLACK_SIGNING_SECRET`

### "Bot doesn't respond to mentions"
1. Make sure the bot is added to the channel (invite it with `/invite @SEO Knowledge Base`)
2. Check that `app_mention` event is subscribed
3. Check the bot logs for errors

### "Slash command not working"
1. Reinstall the app to workspace (OAuth & Permissions > Install App)
2. Make sure the command was saved correctly

### "Socket connection failed"
1. Verify `SLACK_APP_TOKEN` is correct and starts with `xapp-`
2. Ensure Socket Mode is enabled in app settings

### Bot responds slowly
The RAG query can take 3-8 seconds depending on:
- Database size
- OpenAI API latency (embeddings)
- Anthropic API latency (Claude response)

This is normal - the bot shows "Searching..." while processing.

## Running in Background

To keep the bot running after closing your terminal:

### Using nohup (simple)
```bash
nohup python slack_bot.py > slack_bot.log 2>&1 &
```

### Using screen
```bash
screen -S slackbot
python slack_bot.py
# Press Ctrl+A then D to detach
# Reconnect with: screen -r slackbot
```

### Using systemd (production)
Create `/etc/systemd/system/seo-slackbot.service`:
```ini
[Unit]
Description=SEO Knowledge Base Slack Bot
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/seo-knowledge-base
Environment=PATH=/path/to/seo-knowledge-base/venv/bin
EnvironmentFile=/path/to/seo-knowledge-base/.env
ExecStart=/path/to/seo-knowledge-base/venv/bin/python slack_bot.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl enable seo-slackbot
sudo systemctl start seo-slackbot
```

## Security Notes

- Never commit tokens to version control
- Add `.env` to your `.gitignore`
- Rotate tokens if you suspect they've been compromised (in Slack App settings)
- The bot only has access to channels it's invited to
