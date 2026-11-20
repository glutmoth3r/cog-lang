# cogweb API showcase

A small local HTTP server showing `cogweb` routing with named and wildcard
parameters, JSON and HTML responses, request-size limits, escaped route data,
security headers, and async TCP handling.

Run from the repository root:

```sh
cog run examples/libraries/cogweb/showcase
```

Open `http://127.0.0.1:4174/`. The project is `0.1.0` and uses `cogweb`
`0.3.0`. This teaching server handles one connection at a time.

Try `GET /api/status`, `GET /api/hello/Cog`, and
`GET /api/path/examples/libraries` to see JSON, escaped route parameters, and
wildcard matching.
