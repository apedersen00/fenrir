set(FenrirSRC
    fenrir.cpp
)

# Form the full path to the source files...
PREPEND(FenrirSRC)

# ... and pass the variable to the parent scope.
set(FenrirSRC ${FenrirSRC} PARENT_SCOPE)