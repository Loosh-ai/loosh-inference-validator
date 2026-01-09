module.exports = {
  apps: [
    {
      name: 'loosh-inference-validator',
      script: 'uvicorn',
      args: `validator.validator_server:app --host ${process.env.API_HOST || '0.0.0.0'} --port ${process.env.API_PORT || '8000'}`,
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
        NODE_ENV: 'production',
        API_HOST: process.env.API_HOST || '0.0.0.0',
        API_PORT: process.env.API_PORT || '8000'
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

