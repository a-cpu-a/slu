#pragma once

#define LUACC_COL_HEADER "\x1b[38;5;" //ansi color

#define LUACC_DEFAULT       LUACC_COL_HEADER "15m"  // ,[]:.
#define LUACC_STRING_DOUBLE LUACC_COL_HEADER "114m" // "<- ->" string '<- ->'
#define LUACC_STRING_SINGLE LUACC_COL_HEADER "151m" // '<- ->'
#define LUACC_STRING_INNER  LUACC_COL_HEADER "71m"  // "this stuff inside the quotes"
#define LUACC_NUMBER        LUACC_COL_HEADER "173m" // 123.1553, integer, number, zero, double, float
#define LUACC_BRACKET       LUACC_COL_HEADER "221m" // {}
#define LUACC_INVALID       LUACC_COL_HEADER "217m" // not missing invalid unfinished malformed bad error
#define LUACC_FOR           LUACC_COL_HEADER "219m" // for
#define LUACC_FUNCTION      LUACC_COL_HEADER "167m" // function
#define LUACC_ARGUMENT      LUACC_COL_HEADER "159m" // "argument", "arguments"
#define LUACC_BOOLEAN       LUACC_COL_HEADER "81m"  // bool boolean
#define LUACC_NIL           LUACC_COL_HEADER "210m" // nil NaN
#define LUACC_PATH			LUACC_COL_HEADER "251m"
#define LUACC_STACKTRACE	LUACC_PATH


#define LUACC_SINGLE_STRING_STARTER LUACC_STRING_SINGLE "'" 
#define LUACC_START_SINGLE_STRING LUACC_SINGLE_STRING_STARTER LUACC_STRING_INNER
#define LUACC_END_SINGLE_STRING LUACC_SINGLE_STRING_STARTER LUACC_DEFAULT

#define LUACC_SINGLE_STRING_INCOL(_COL,_TEXT) LUACC_STRING_SINGLE "'" _COL _TEXT LUACC_END_SINGLE_STRING
#define LUACC_SINGLE_STRING(_TEXT) LUACC_SINGLE_STRING_INCOL(LUACC_STRING_INNER,_TEXT)

#define LUACC_COL(_COL,_TEXT) _COL _TEXT LUACC_DEFAULT 
#define LUACC_NUM_COL(_TEXT) LUACC_COL(LUACC_NUMBER,_TEXT) 

//Order: len, letter, caps

#define LC_string LUACC_COL(LUACC_STRING_DOUBLE,"string")

#define LC_double LUACC_COL(LUACC_NUMBER,"double")
#define LC_Integer LUACC_COL(LUACC_NUMBER,"Integer")
#define LC_integer LUACC_COL(LUACC_NUMBER,"integer")
#define LC_number LUACC_COL(LUACC_NUMBER,"number")
#define LC_Number LUACC_COL(LUACC_NUMBER,"Number")
#define LC_zero LUACC_COL(LUACC_NUMBER,"zero")

#define LC_error LUACC_COL(LUACC_INVALID,"error")
#define LC_failed LUACC_COL(LUACC_INVALID,"failed")
#define LC_Invalid LUACC_COL(LUACC_INVALID,"Invalid")
#define LC_invalid LUACC_COL(LUACC_INVALID,"invalid")
#define LC_not LUACC_COL(LUACC_INVALID,"not")
#define LC_unfinished LUACC_COL(LUACC_INVALID,"unfinished")

#define LC_for LUACC_COL(LUACC_FOR,"for")
#define LC_Function LUACC_COL(LUACC_FUNCTION,"Function")
#define LC_function LUACC_COL(LUACC_FUNCTION,"function")

#define LC_argument LUACC_COL(LUACC_ARGUMENT,"argument")
#define LC_arguments LUACC_COL(LUACC_ARGUMENT,"arguments")