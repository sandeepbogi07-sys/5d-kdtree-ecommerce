# Threadly — 5D KD-Tree Clothing Store (Database-Free)

A Flask + HTML/CSS/JavaScript clothing-store prototype using a custom five-dimensional KD-Tree for nearest-neighbour product recommendations. It runs without SQLite or any external database.

## Run on Windows (VS Code)

1. Extract the ZIP and open the `5d-kdtree-ecommerce` folder in VS Code.
2. Open **Terminal → New Terminal**.
3. Run:

   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   python -m pip install -r requirements.txt
   python backend/app.py
   ```

4. Open **http://127.0.0.1:5000** in your browser.
5. To run tests, install requirements and run `python -m pytest backend/tests` from the project root.

`run.bat` is also included as a Windows shortcut. The application serves its frontend and API from the same Flask process; no separate frontend server is required.

## Included features

- Modern responsive Threadly storefront and generated garment illustrations.
- Search, category, price, GSM, stretch, colour, rating and stock filters.
- KD-Tree-ranked sorting and nearest-neighbour recommendations.
- Cart with quantity controls and subtotal; wishlist; comparison of up to three products.
- KD-Tree metrics, a live tree-structure explorer (first 31 nodes), and benchmark comparison against brute force.
- Input validation on product writes and search vectors, duplicate-name prevention, and development admin-token protection.
- Automated tests for KD-Tree behavior, API flows, validation and the visualization endpoint.

## Data and limitations

Products are generated in memory at startup. Product changes, cart contents, wishlist and comparison selections are temporary and reset when the server/browser session is restarted or refreshed. The cart is a demonstration only: there is no checkout, payment processing, user account, or order persistence. Product art is locally generated SVG-style illustration rather than real product photography. Google Fonts is optional; the page falls back to system fonts if offline.

## KD-Tree model

Each product is encoded as `[normalized price, normalized GSM, normalized stretch, cos(hue), sin(hue)]`. The hue representation handles the circular nature of colour. The custom KD-Tree uses median splits and weighted Euclidean distance, with branch pruning for k-nearest-neighbour and radius searches.

## API overview

- `GET /api/health`
- `GET /api/products`, `GET /api/products/<id>`
- `GET /api/recommendations/<id>?k=5&compare=1`
- `POST /api/search/knn`, `POST /api/search/radius`
- `GET /api/kdtree/stats`, `GET /api/kdtree/visualization`
- `GET /api/benchmark`
- Admin-only writes: `POST /api/products`, `PUT /api/products/<id>`, `DELETE /api/products/<id>`, `POST /api/kdtree/rebuild`

Admin writes require `X-Admin-Token`. The fallback `dev-admin-token` is for local development only. Set the `ADMIN_TOKEN` environment variable to a strong private value before any shared deployment. Do not expose the development server directly to the public internet.
