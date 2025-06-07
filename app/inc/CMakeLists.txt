set(FenrirINC
    fenrir.hpp
)

# Form the full path to the source files...
PREPEND(FenrirINC)

# ... and pass the variable to the parent scope.
set(FenrirINC ${FenrirINC} PARENT_SCOPE)