<<<<<<< HEAD
// Fix for Node.js compatibility
if (typeof File === 'undefined') {
    global.File = class File {
        constructor() {
            // Mock File class for compatibility
        }
    };
}

=======
>>>>>>> parent of 42baa98 (Fix Node.js and Python errors)
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const axios = require('axios');
const cheerio = require('cheerio');

const app = express();
const PORT = process.env.PORT || 3000;
const API_CALLBACK_URL = process.env.API_CALLBACK_URL || 'http://localhost:8001/api-callback';

// Middleware
app.use(helmet()); // Security headers
app.use(cors()); // Enable CORS
app.use(express.json({ limit: '10mb' })); // Parse JSON bodies
app.use(express.urlencoded({ extended: true })); // Parse URL-encoded bodies

// Store webhook data in memory (in production, use a database)
const webhookData = [];
const extractedData = [];

// Helper function to send callback to API
async function sendCallbackToAPI(extractedContent, webhookPayload) {
  try {
    const requestId = webhookPayload.body?.request_id || 
                     webhookPayload.body?.id || 
                     webhookPayload.body?.job_id;
    
    if (!requestId) {
      console.log('⚠️ No request_id found in webhook payload, skipping API callback');
      return;
    }
    
    const callbackData = {
      request_id: requestId,
      success: extractedContent.type !== 'error',
      markdown_content: extractedContent.type === 'text' ? extractedContent.content : 
                       extractedContent.type === 'html' ? extractedContent.text : 
                       extractedContent.type === 'json' ? JSON.stringify(extractedContent.data) : 
                       extractedContent.content || '',
      extracted_data: extractedContent,
      timestamp: new Date().toISOString()
    };
    
    console.log(`📤 Sending callback to API for request_id: ${requestId}`);
    
    const response = await axios.post(API_CALLBACK_URL, callbackData, {
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    if (response.status === 200) {
      console.log(`✅ Successfully sent callback to API for request_id: ${requestId}`);
    } else {
      console.log(`⚠️ API callback returned status ${response.status} for request_id: ${requestId}`);
    }
    
  } catch (error) {
    console.error(`❌ Failed to send callback to API:`, error.message);
  }
}

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    timestamp: new Date().toISOString(),
    webhook_data_count: webhookData.length,
    extracted_data_count: extractedData.length
  });
});

// Get all webhook data
app.get('/webhooks', (req, res) => {
  res.json({
    webhooks: webhookData,
    extracted: extractedData,
    total_webhooks: webhookData.length,
    total_extracted: extractedData.length
  });
});

// Clear all data
app.delete('/webhooks', (req, res) => {
  webhookData.length = 0;
  extractedData.length = 0;
  res.json({ message: 'All webhook data cleared' });
});

// Main webhook endpoint
app.post('/webhook', async (req, res) => {
  try {
    const webhookPayload = {
      headers: req.headers,
      body: req.body,
      timestamp: new Date().toISOString(),
      ip: req.ip || req.connection.remoteAddress
    };
    
    webhookData.push(webhookPayload);
    
    console.log('📨 Received webhook:', {
      timestamp: webhookPayload.timestamp,
      headers: webhookPayload.headers,
      body: webhookPayload.body
    });
    
    // Check if this is a datalab completion webhook
    const isDatalabWebhook = req.body?.status === 'completed' || 
                            req.body?.status === 'complete' || 
                            req.body?.success === true ||
                            (req.body?.markdown && req.body?.markdown.trim().length > 0);
    
    if (isDatalabWebhook) {
      console.log('🎯 Detected datalab processing completion webhook');
      
      const markdownContent = req.body?.markdown || req.body?.content || req.body?.result;
      
      if (markdownContent) {
        const extractedContent = {
          type: 'text',
          content: markdownContent,
          url: 'datalab_webhook',
          timestamp: new Date().toISOString()
        };
        
        extractedData.push(extractedContent);
        console.log(`✅ Extracted markdown content from datalab webhook (${markdownContent.length} chars)`);
        
        await sendCallbackToAPI(extractedContent, webhookPayload);
      }
    }
    
    // Check if this is a general content extraction webhook
    if (req.body?.url && (req.body?.text || req.body?.html || req.body?.markdown)) {
      console.log('🎯 Detected content extraction webhook');
      
      const extractedContent = {
        type: req.body?.type || 'text',
        content: req.body?.text || req.body?.html || req.body?.markdown,
        url: req.body?.url,
        timestamp: new Date().toISOString()
      };
      
      extractedData.push(extractedContent);
      console.log(`✅ Extracted content from webhook (${extractedContent.content.length} chars)`);
      
      await sendCallbackToAPI(extractedContent, webhookPayload);
    }
    
    res.json({ 
      success: true, 
      message: 'Webhook received and processed',
      timestamp: webhookPayload.timestamp
    });
    
  } catch (error) {
    console.error('❌ Webhook processing error:', error);
    res.status(500).json({ 
      success: false, 
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Start server
app.listen(PORT, '0.0.0.0', () => {
  console.log(`🚀 Webhook server running on port ${PORT}`);
  console.log(`📡 API callback URL: ${API_CALLBACK_URL}`);
  console.log(`🔗 Webhook endpoint: http://0.0.0.0:${PORT}/webhook`);
  console.log(`❤️  Health check: http://0.0.0.0:${PORT}/health`);
});
