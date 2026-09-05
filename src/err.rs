//! Central error handling for the C ABI: error codes + a thread-local last-error
//! message. Kept separate so every layer (tensor/session/rtw/rule) reports the
//! same way.

use std::cell::RefCell;

/// Error codes matching `include/rtorch_api.h`.
pub const RTORCH_API_OK: i32 = 0;
pub const RTORCH_API_E_PARAM: i32 = 1;
pub const RTORCH_API_E_NOMEM: i32 = 2;
pub const RTORCH_API_E_RT: i32 = 3;
pub const RTORCH_API_E_IO: i32 = 4;
pub const RTORCH_API_E_PARSE: i32 = 5;

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

/// Overwrite the thread-local last-error message.
pub fn set_last_error(msg: &str) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = msg.to_string();
    });
}

/// Read the current last-error message (valid until the next set).
pub fn last_error() -> String {
    LAST_ERROR.with(|e| e.borrow().clone())
}

/// A convenience: report an error string and return code.
pub fn err(code: i32, msg: &str) -> i32 {
    set_last_error(msg);
    code
}
