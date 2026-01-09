import copyreg
import socket
import ssl

def _rebuild_socket(family, sock_type, proto, timeout):
    s = socket.socket(family, sock_type, proto)
    s.settimeout(timeout)
    return s

def _reduce_socket(s: socket.socket):
    return _rebuild_socket, (s.family, s.type, s.proto, s.gettimeout())

def _rebuild_ssl_context(protocol, check_hostname, verify_mode, options,
                         minimum_version, maximum_version, verify_flags):
    ctx = ssl.SSLContext(protocol)
    ctx.check_hostname = check_hostname
    ctx.verify_mode = verify_mode
    ctx.options = options
    if minimum_version is not None and hasattr(ctx, "minimum_version"):
        ctx.minimum_version = minimum_version
    if maximum_version is not None and hasattr(ctx, "maximum_version"):
        ctx.maximum_version = maximum_version
    if verify_flags is not None and hasattr(ctx, "verify_flags"):
        ctx.verify_flags = verify_flags
    return ctx

def _reduce_ssl_context(ctx: ssl.SSLContext):
    return _rebuild_ssl_context, (
        ctx.protocol,
        ctx.check_hostname,
        ctx.verify_mode,
        ctx.options,
        getattr(ctx, "minimum_version", None),
        getattr(ctx, "maximum_version", None),
        getattr(ctx, "verify_flags", None),
    )

def register_dill_reducers():
    # ✅ 只注册“相对可重建”的两类
    copyreg.pickle(socket.socket, _reduce_socket)
    copyreg.pickle(ssl.SSLContext, _reduce_ssl_context)

