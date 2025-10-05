echo "ERRORS:"
cat nohup.out | grep ERROR
echo ""
echo "PROCESS:"
ps aux | grep sd_ass_bot.py