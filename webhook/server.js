// Fix for Node.js compatibility - MUST be first
if (typeof File === 'undefined') {
    global.File = class File {
        constructor() {
            // Mock File class for compatibility
        }
    };
}

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

// Store pending requests waiting for webhook callbacks
const pendingRequests = new Map();

// Store mapping between API request_id and datalab request_id
const requestIdMapping = new Map();

// Simple webhook server - no authentication needed

// Helper function to send callback to API
async function sendCallbackToAPI(extractedContent, webhookPayload) {
  try {
    // Look for request_id in the webhook payload
    let requestId = webhookPayload.body?.request_id || 
                   webhookPayload.body?.id || 
                   webhookPayload.body?.job_id;
    
    // If this is a datalab completion webhook, extract the datalab request_id
    if (webhookPayload.body?.request_check_url) {
      const urlMatch = webhookPayload.body.request_check_url.match(/\/marker\/([^/?]+)/);
      if (urlMatch) {
        requestId = urlMatch[1];
        console.log(`🔍 Using datalab request_id: ${requestId}`);
      }
    }
    
    if (!requestId) {
      console.log('⚠️ No request_id found in webhook payload, skipping API callback');
      return;
    }
    
    // Prepare callback data
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
    
    // Send callback to API
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

// Helper function to extract data from URL
async function extractDataFromUrl(url) {
  try {
    console.log(`🔍 Extracting data from URL: ${url}`);
    
    const headers = {
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    };
    
    // Add API key to headers if found in URL
    if (url.includes('api_key=')) {
      const apiKeyMatch = url.match(/[?&]api_key=([^&]+)/);
      if (apiKeyMatch) {
        const apiKey = apiKeyMatch[1];
        headers['Authorization'] = `Bearer ${apiKey}`;
        headers['X-API-Key'] = apiKey;
        console.log(`🔑 Added API key to headers`);
      }
    }
    
    const response = await axios.get(url, {
      timeout: 10000,
      headers: headers
    });

    const contentType = response.headers['content-type'] || '';
    let extractedContent = {};

    if (contentType.includes('application/json')) {
      // Handle JSON data
      extractedContent = {
        type: 'json',
        data: response.data,
        url: url,
        timestamp: new Date().toISOString()
      };
    } else if (contentType.includes('text/html')) {
      // Handle HTML data
      const $ = cheerio.load(response.data);
      extractedContent = {
        type: 'html',
        title: $('title').text().trim(),
        description: $('meta[name="description"]').attr('content') || '',
        text: $('body').text().replace(/\s+/g, ' ').trim(),
        links: $('a[href]').map((i, el) => $(el).attr('href')).get(),
        images: $('img[src]').map((i, el) => $(el).attr('src')).get(),
        url: url,
        timestamp: new Date().toISOString()
      };
    } else if (contentType.includes('text/plain') || contentType.includes('text/markdown')) {
      // Handle plain text or markdown
      extractedContent = {
        type: 'text',
        content: response.data,
        url: url,
        timestamp: new Date().toISOString()
      };
    } else {
      // Handle other content types
      extractedContent = {
        type: 'other',
        contentType: contentType,
        content: response.data,
        url: url,
        timestamp: new Date().toISOString()
      };
    }

    return extractedContent;
  } catch (error) {
    console.error(`❌ Error extracting data from ${url}:`, error.message);
    return {
      type: 'error',
      error: error.message,
      url: url,
      timestamp: new Date().toISOString()
    };
  }
}

// Health check endpoint
app.get('/', (req, res) => {
  res.json({
    message: 'Simple Webhook Server is running!',
    status: 'healthy',
    timestamp: new Date().toISOString(),
    totalWebhooks: webhookData.length
  });
});

// Main webhook endpoint - accepts POST requests
app.post('/webhook', async (req, res) => {
  try {
    const webhookPayload = {
      id: Date.now().toString(),
      timestamp: new Date().toISOString(),
      headers: req.headers,
      body: req.body,
      query: req.query,
      method: req.method,
      url: req.url,
      ip: req.ip || req.connection.remoteAddress
    };

    // Store the webhook data
    webhookData.push(webhookPayload);

    // Log the webhook data
    console.log('📨 Received webhook:', {
      id: webhookPayload.id,
      timestamp: webhookPayload.timestamp,
      headers: webhookPayload.headers,
      body: webhookPayload.body
    });

    // Check if webhook contains URLs to extract data from
    const urlsToExtract = [];
    
    // Look for URLs in the webhook body
    if (req.body && typeof req.body === 'object') {
      const bodyStr = JSON.stringify(req.body);
      const urlRegex = /https?:\/\/[^\s"<>]+/g;
      const foundUrls = bodyStr.match(urlRegex);
      if (foundUrls) {
        urlsToExtract.push(...foundUrls);
      }
      
      // Auto-add API key to datalab.to URLs
      const datalabApiKey = 'QP4dh9aa9gzIbKxJ0Xj9Rz0anAyqYxGgIajQqvuGDhA';
      urlsToExtract.forEach((url, index) => {
        if (url.includes('datalab.to') && !url.includes('api_key=')) {
          const separator = url.includes('?') ? '&' : '?';
          urlsToExtract[index] = `${url}${separator}api_key=${datalabApiKey}&key=${datalabApiKey}&token=${datalabApiKey}`;
          console.log(`🔑 Auto-added API key to datalab.to URL`);
        }
      });
    }
    
    // Check if this is a datalab processing completion webhook
    const isDatalabWebhook = req.body?.status === 'completed' || 
                            req.body?.status === 'complete' || 
                            req.body?.success === true ||
                            (req.body?.markdown && req.body?.markdown.trim().length > 0) ||
                            req.body?.request_check_url; // Check for datalab completion notification
    
    if (isDatalabWebhook) {
      console.log('🎯 Detected datalab processing completion webhook');
      
      // Extract datalab request_id from the request_check_url
      let datalabRequestId = null;
      if (req.body?.request_check_url) {
        const urlMatch = req.body.request_check_url.match(/\/marker\/([^/?]+)/);
        if (urlMatch) {
          datalabRequestId = urlMatch[1];
          console.log(`🔍 Found datalab request_id: ${datalabRequestId}`);
        }
      }
      
      // First, try to extract markdown content directly from webhook body
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
        
        // Send callback to API
        await sendCallbackToAPI(extractedContent, webhookPayload);
      } else if (req.body?.request_check_url) {
        // If no direct content, fetch from the request_check_url
        console.log(`🔍 Fetching content from request_check_url: ${req.body.request_check_url}`);
        
        try {
          const extractedContent = await extractDataFromUrl(req.body.request_check_url);
          
          if (extractedContent.type !== 'error') {
            extractedData.push(extractedContent);
            console.log(`✅ Successfully fetched content from datalab API`);
            
            // Send callback to API
            await sendCallbackToAPI(extractedContent, webhookPayload);
          } else {
            console.log(`❌ Failed to fetch content from datalab API: ${extractedContent.error}`);
          }
        } catch (error) {
          console.error(`❌ Error fetching from request_check_url:`, error.message);
        }
      }
    }

    // Extract data from URLs if found
    let extractedResults = [];
    if (urlsToExtract.length > 0) {
      console.log(`🔍 Found ${urlsToExtract.length} URLs to extract data from`);
      
      for (const url of urlsToExtract) {
        try {
          const extractedContent = await extractDataFromUrl(url);
          extractedResults.push(extractedContent);
          extractedData.push(extractedContent);
          
          console.log(`✅ Successfully extracted data from: ${url}`);
          
          // Display extracted data in terminal
          console.log('\n' + '='.repeat(80));
          console.log(`📄 EXTRACTED DATA FROM: ${url}`);
          console.log('='.repeat(80));
          console.log(`Type: ${extractedContent.type}`);
          console.log(`Timestamp: ${extractedContent.timestamp}`);
          console.log('-'.repeat(40));
          
          if (extractedContent.type === 'html') {
            console.log(`Title: ${extractedContent.title}`);
            console.log(`Description: ${extractedContent.description}`);
            console.log(`\nText Content (first 500 chars):`);
            console.log(extractedContent.text.substring(0, 500) + '...');
            console.log(`\nLinks found: ${extractedContent.links.length}`);
            console.log(`Images found: ${extractedContent.images.length}`);
          } else if (extractedContent.type === 'json') {
            console.log('JSON Data:');
            console.log(JSON.stringify(extractedContent.data, null, 2));
          } else if (extractedContent.type === 'text') {
            console.log('Text Content:');
            console.log(extractedContent.content.substring(0, 1000) + '...');
          } else if (extractedContent.type === 'error') {
            console.log(`❌ Error: ${extractedContent.error}`);
          } else {
            console.log('Raw Content:');
            console.log(JSON.stringify(extractedContent, null, 2));
          }
          
          console.log('='.repeat(80) + '\n');
          
          // Send callback to API if this is a datalab processing result
          await sendCallbackToAPI(extractedContent, webhookPayload);
          
        } catch (error) {
          console.error(`❌ Failed to extract data from ${url}:`, error.message);
        }
      }
    }

    // Respond with success
    res.status(200).json({
      success: true,
      message: 'Webhook received successfully',
      webhookId: webhookPayload.id,
      urlsFound: urlsToExtract.length
    });

  } catch (error) {
    console.error('❌ Error processing webhook:', error);
    res.status(500).json({
      success: false,
      message: 'Internal server error',
      error: error.message
    });
  }
});

// Get all webhooks endpoint
app.get('/webhooks', (req, res) => {
  res.json({
    success: true,
    count: webhookData.length,
    webhooks: webhookData
  });
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
});

// Catch-all for undefined routes
app.use('*', (req, res) => {
  res.status(404).json({
    success: false,
    message: 'Route not found',
    availableEndpoints: [
      'GET / - Health check',
      'POST /webhook - Receive webhook and extract data from URLs',
      'GET /webhooks - List all webhooks',
      'GET /health - Health check'
    ]
  });
});

// Error handling middleware
app.use((error, req, res, next) => {
  console.error('❌ Unhandled error:', error);
  res.status(500).json({
    success: false,
    message: 'Internal server error',
    error: process.env.NODE_ENV === 'production' ? 'Something went wrong' : error.message
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Webhook server running on port ${PORT}`);
  console.log(`📡 API callback URL: ${API_CALLBACK_URL}`);
  console.log(`🔗 Webhook endpoint: http://0.0.0.0:${PORT}/webhook`);
  console.log(`❤️  Health check: http://0.0.0.0:${PORT}/health`);
});

module.exports = app;