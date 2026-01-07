# Set prompt colors based on container name
case "${CONTAINER_NAME:-container}" in
    miner1)
        COLOR='\033[1;32m'  # Bold green
        ;;
    miner2)
        COLOR='\033[1;36m'  # Bold cyan
        ;;
    miner3)
        COLOR='\033[1;33m'  # Bold yellow
        ;;
    validator)
        COLOR='\033[1;35m'  # Bold magenta
        ;;
    challenge-api)
        COLOR='\033[1;31m'  # Bold red
        ;;
    *)
        COLOR='\033[1;34m'  # Bold blue (default)
        ;;
esac

export PS1="\[${COLOR}\][${CONTAINER_NAME:-container}]\[\033[0m\] \h:\w \$ "
