module.exports = {
  apps: [
    {
      name: 'loosh-inference-validator',
      script: 'validator/main.py',
      interpreter: process.env.PYTHON_INTERPRETER || 'python3',
      cwd: process.env.VALIDATOR_WORKDIR || process.cwd(),
      watch: false,
      autorestart: true,
      max_restarts: 10,
      min_uptime: '10s',
      max_memory_restart: '2G',
      env: {
        PYTHONPATH: process.env.VALIDATOR_WORKDIR || process.cwd(),
        PYTHONUNBUFFERED: '1',
        NODE_ENV: 'production'
      },
      error_file: './logs/validator-error.log',
      out_file: './logs/validator-out.log',
      log_file: './logs/validator-combined.log',
      time: true,
      merge_logs: true,
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z'
    }
  ]
};

