#!/bin/bash
# Miner service management script

echo "🎯 PRECOG MINER SERVICE MANAGER"
echo "================================"

case "$1" in
    start)
        echo "🚀 Starting miner service..."
        systemctl --user start precog-miner.service
        ;;
    stop)
        echo "🛑 Stopping miner service..."
        systemctl --user stop precog-miner.service
        ;;
    restart)
        echo "🔄 Restarting miner service..."
        systemctl --user restart precog-miner.service
        ;;
    status)
        echo "📊 Service Status:"
        systemctl --user status precog-miner.service --no-pager -l
        ;;
    logs)
        echo "📋 Recent Logs:"
        journalctl --user -u precog-miner.service -n 20 --no-pager
        ;;
    enable)
        echo "✅ Enabling service to start on login..."
        systemctl --user enable precog-miner.service
        ;;
    disable)
        echo "❌ Disabling service..."
        systemctl --user disable precog-miner.service
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|enable|disable}"
        echo ""
        echo "Examples:"
        echo "  $0 status    # Check if running"
        echo "  $0 logs      # View recent logs"
        echo "  $0 restart   # Restart service"
        echo "  $0 stop      # Stop service"
        ;;
esac
